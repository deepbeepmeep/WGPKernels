// Copyright (c) 2026 DeepBeepMeep - Apache License 2.0 (see LICENSE.txt)
// INT8 weight GEMMs for Quanto INT8 and ConvRot checkpoints: Y = X @ (W * sw).T + bias.
// W is the checkpoint's INT8 (N, K) matrix with FP32 per-output-channel scales; it is never dequantized or copied.
#if defined(__HIP__)
// AMD (HIP, RDNA4 gfx12, wave32): the shared code keeps CUDA names, mapped here; the tensor-core kernels (gemm_s8, w8a16_small)
// have AMD counterparts behind the same C API (gemm_s8_wmma; the few-token GEMV for up to 32 tokens).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <rocwmma/rocwmma.hpp>
#define WGP_AMD 1
using __nv_bfloat16 = __hip_bfloat16;
using __nv_bfloat162 = __hip_bfloat162;
#define cudaStream_t hipStream_t
#define cudaGetLastError hipGetLastError
#define cudaSuccess hipSuccess
#define cudaSetDevice hipSetDevice
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define __ldcg(p) (*(p))
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#define WGP_AMD 0
#endif
#include <stdint.h>
#include <type_traits>

#define DEVICE __device__ __forceinline__

// Butterfly exchange within a 32-lane warp (an RDNA wave32).
DEVICE float shfl_xor(float v, int lane_mask) {
#if WGP_AMD
    return __shfl_xor(v, lane_mask, 32);
#else
    return __shfl_xor_sync(0xffffffffu, v, lane_mask);
#endif
}

#if !WGP_AMD
// ---------------------------------------------------------------------------------------------------------------------
// PTX helpers
// ---------------------------------------------------------------------------------------------------------------------
DEVICE uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

// Turing (sm_75) has no asynchronous copies and no m16n8k32 INT8 MMA: copies are synchronous 16-byte loads and stores (the
// pipeline's waits are no-ops, its barriers keep the order), and each m16n8k32 is four m8n8k16 on the same fragment registers
// (exact INT32 sums: the results are the same).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#define WGP_SM75 1
#else
#define WGP_SM75 0
#endif

DEVICE void cp_async16(uint32_t dst, const void* src, bool valid) {
    // Zero fill out-of-range chunks: partial M/N/K tiles need no separate code path.
#if WGP_SM75
    const uint4 v = valid ? __ldcg(reinterpret_cast<const uint4*>(src)) : make_uint4(0, 0, 0, 0);
    asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(dst), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
#else
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(valid ? 16 : 0));
#endif
}
// 4-byte copy (per-row scales): .ca, the cp.async form that takes 4 bytes.
DEVICE void cp_async4(uint32_t dst, const void* src, bool valid) {
#if WGP_SM75
    const uint32_t v = valid ? __ldca(reinterpret_cast<const unsigned int*>(src)) : 0u;
    asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(dst), "r"(v));
#else
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(dst), "l"(src), "r"(valid ? 4 : 0));
#endif
}
#if WGP_SM75
DEVICE void cp_async_commit() {}
template<int N> DEVICE void cp_async_wait() {}
#else
DEVICE void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template<int N> DEVICE void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N > 0 ? N : 0)); }  // Turing's 2-stage tiles: never run here
#endif

DEVICE void ldmatrix_x4(uint32_t* r, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

DEVICE void mma_s8(int* c, const uint32_t* a, const uint32_t* b) {
#if WGP_SM75
    // a0/a1: rows g / g + 8 of k 0-15, a2/a3: of k 16-31; b0/b1: k 0-15 / 16-31 (the m16n8k32 fragments of these thread positions).
    // Separate statements without volatile: the compiler interleaves the MMAs of independent accumulators.
    #pragma unroll
    for (int kh = 0; kh < 2; ++kh)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
                : "+r"(c[2 * h]), "+r"(c[2 * h + 1]) : "r"(a[h + 2 * kh]), "r"(b[kh]));
#else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}
#endif

// Block sums start at BLOCK_BIAS, the bits of 1.5 * 2^23: the exact INT32 additions of the MMAs (|block sum| <= 128 * 128 * 128 < 2^22)
// leave the float 1.5 * 2^23 + sum, so a float subtraction recovers the sum exactly (no integer add, no slow I2F).
constexpr int BLOCK_BIAS = 0x4B400000;

// accf[0..1] += scale * (block sums of the pair), then the sums restart at the bias.
DEVICE void fold_block_pair(float* accf, int* sums, float scale) {
    accf[0] = fmaf(__int_as_float(sums[0]) - 12582912.f, scale, accf[0]);
    accf[1] = fmaf(__int_as_float(sums[1]) - 12582912.f, scale, accf[1]);
    sums[0] = sums[1] = BLOCK_BIAS;
}

template<class T> DEVICE float to_float(T v) { return float(v); }
template<> DEVICE float to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template<> DEVICE float to_float<__half>(__half v) { return __half2float(v); }

template<class T> struct Pack2;
#if WGP_AMD
template<> struct Pack2<__nv_bfloat16> { static DEVICE uint32_t make(float a, float b) {
    __nv_bfloat16 lo = __float2bfloat16(a), hi = __float2bfloat16(b);
    return uint32_t(*reinterpret_cast<uint16_t*>(&lo)) | uint32_t(*reinterpret_cast<uint16_t*>(&hi)) << 16;
} };
#else
template<> struct Pack2<__nv_bfloat16> { static DEVICE uint32_t make(float a, float b) { __nv_bfloat162 v = __floats2bfloat162_rn(a, b); return *reinterpret_cast<uint32_t*>(&v); } };
#endif
template<> struct Pack2<__half> { static DEVICE uint32_t make(float a, float b) { __half2 v = __floats2half2_rn(a, b); return *reinterpret_cast<uint32_t*>(&v); } };

// ---------------------------------------------------------------------------------------------------------------------
// Row quantization: optional ConvRot rotation, one FP32 scale per row, INT8 values (and an optional INT8 residual "lo"
// such that x ~= scale * (hi + lo / 254)). One CTA per row; the row is read twice (the second read hits L1/L2) so that
// registers do not limit K.
// ---------------------------------------------------------------------------------------------------------------------
// Lane l of a warp holds elements 8l..8l+7 of a 256-element chunk: base-4 digits d0 = j&3, d1 = (j>>2)|((l&1)<<1),
// d2 = (l>>1)&3, d3 = (l>>3)&3. ConvRot's regular H4 is out[d] = sum - 2 * in[3 - d] along each digit.
template<int GROUP> DEVICE void rotate_chunk(float* v, int lane) {
    if (GROUP >= 4) {
        #pragma unroll
        for (int h = 0; h < 8; h += 4) {
            float s = (v[h] + v[h + 1]) + (v[h + 2] + v[h + 3]);
            float a = v[h], b = v[h + 1], c = v[h + 2], d = v[h + 3];
            v[h] = s - 2.f * d; v[h + 1] = s - 2.f * c; v[h + 2] = s - 2.f * b; v[h + 3] = s - 2.f * a;
        }
    }
    if (GROUP >= 16) {
        float out[8];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float t = v[j] + v[j + 4];
            float s = t + shfl_xor(t, 1);
            // Digit 3 - d lives in the partner lane at the element with the other low digit bit.
            out[j] = s - 2.f * shfl_xor(v[j + 4], 1);
            out[j + 4] = s - 2.f * shfl_xor(v[j], 1);
        }
        #pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = out[j];
    }
    #pragma unroll
    for (int stage = 0; stage < 2; ++stage) {
        if ((stage == 0 && GROUP >= 64) || (stage == 1 && GROUP >= 256)) {
            const int lo = stage == 0 ? 2 : 8;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float a = v[j] + shfl_xor(v[j], lo);
                float s = a + shfl_xor(a, 2 * lo);
                v[j] = s - 2.f * shfl_xor(v[j], 3 * lo);
            }
        }
    }
}

template<class T> DEVICE void load8(const T* p, float* v, bool valid) {
    if constexpr (sizeof(T) == 2) {
        uint4 raw = valid ? *reinterpret_cast<const uint4*>(p) : make_uint4(0, 0, 0, 0);
        const T* e = reinterpret_cast<const T*>(&raw);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = to_float(e[j]);
    } else {
        float4 a = valid ? reinterpret_cast<const float4*>(p)[0] : make_float4(0, 0, 0, 0);
        float4 b = valid ? reinterpret_cast<const float4*>(p)[1] : make_float4(0, 0, 0, 0);
        v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
    }
}

DEVICE uint32_t pack_s8x4(int a, int b, int c, int d) {
    return (uint32_t(a) & 0xff) | ((uint32_t(b) & 0xff) << 8) | ((uint32_t(c) & 0xff) << 16) | (uint32_t(d) << 24);
}

// Input operations applied in FP32 where the activation is quantized (or rotated for the few-token paths), so that the transformed
// activation is never rounded or stored: ACT_RMS = RMSNorm x / sqrt(mean(x^2) + eps) * weight over the K inputs of a row;
// ACT_SWIGLU = silu(gate) * value of a row holding [gate (K) | value (K)].
enum { ACT_NONE = 0, ACT_RMS = 1, ACT_SWIGLU = 2 };

template<class T, int ACT> DEVICE void load_act(const T* row, int k, int col, float* v, bool valid) {
    load8(row + col, v, valid);
    if (ACT == ACT_SWIGLU) {
        float u[8];
        load8(row + k + col, u, valid);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = v[j] / (1.f + expf(-v[j])) * u[j];
    }
}

// x * inv_rms * weight for the 8 elements at col (weight optional).
template<class T> DEVICE void apply_rms(float* v, float inv, const T* weight, int col, bool valid) {
    float w[8];
    if (weight) load8(weight + col, w, valid);
    #pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = v[j] * inv * (weight ? w[j] : 1.f);
}

DEVICE float rms_inverse(float sum_squares, int k, float eps) { return 1.f / sqrtf(sum_squares / float(k) + eps); }

// Sum over the CTA's 8 warps in a fixed order (deterministic).
DEVICE float cta_sum(float v, float* shared8, int lane, int warp) {
    #pragma unroll
    for (int d = 16; d; d >>= 1) v += shfl_xor(v, d);
    if (!lane) shared8[warp] = v;
    __syncthreads();
    float s = shared8[0];
    #pragma unroll
    for (int w = 1; w < 8; ++w) s += shared8[w];
    return s;
}

// 1 / RMS of each row (RMSNorm rows longer than the register kernels hold): one CTA per row.
template<class T, int ACT>
__global__ void __launch_bounds__(256) row_inv_rms(const T* __restrict__ x, float* __restrict__ inv_rms, int k, int64_t ldx, float eps) {
    __shared__ float warp_ss[8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const T* row = x + int64_t(blockIdx.x) * ldx;
    float ss = 0.f;
    for (int base = warp * 256; base < k; base += 8 * 256) {
        float v[8];
        load_act<T, ACT>(row, k, base + lane * 8, v, base + lane * 8 < k);
        #pragma unroll
        for (int j = 0; j < 8; ++j) ss = fmaf(v[j], v[j], ss);
    }
    ss = cta_sum(ss, warp_ss, lane, warp);
    if (!threadIdx.x) inv_rms[blockIdx.x] = rms_inverse(ss, k, eps);
}

template<class T, int GROUP, bool HILO, int ACT>
__global__ void __launch_bounds__(256) quantize_rows(const T* __restrict__ x, int8_t* __restrict__ hi, int8_t* __restrict__ lo,
                                                     float* __restrict__ scale, int k, int64_t ldx, const T* __restrict__ act_weight, float eps) {
    __shared__ float warp_max[8], warp_ss[8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const T* row = x + int64_t(blockIdx.x) * ldx;
    float inv_rms = 1.f;
    if (ACT == ACT_RMS) {
        float ss = 0.f;
        for (int base = warp * 256; base < k; base += 8 * 256) {
            float v[8];
            load8(row + base + lane * 8, v, base + lane * 8 < k);
            #pragma unroll
            for (int j = 0; j < 8; ++j) ss = fmaf(v[j], v[j], ss);
        }
        inv_rms = rms_inverse(cta_sum(ss, warp_ss, lane, warp), k, eps);
    }
    float amax = 0.f;
    for (int base = warp * 256; base < k; base += 8 * 256) {
        float v[8];
        load_act<T, ACT>(row, k, base + lane * 8, v, base + lane * 8 < k);
        if (ACT == ACT_RMS) apply_rms(v, inv_rms, act_weight, base + lane * 8, base + lane * 8 < k);
        rotate_chunk<GROUP>(v, lane);
        #pragma unroll
        for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(v[j]));
    }
    #pragma unroll
    for (int d = 16; d; d >>= 1) amax = fmaxf(amax, shfl_xor(amax, d));
    if (!lane) warp_max[warp] = amax;
    __syncthreads();
    amax = warp_max[0];
    #pragma unroll
    for (int w = 1; w < 8; ++w) amax = fmaxf(amax, warp_max[w]);
    // Unnormalized rotation: fold 1/sqrt(GROUP) (a power of two) into the stored scale.
    const float norm = GROUP == 256 ? 1.f / 16 : GROUP == 64 ? 1.f / 8 : GROUP == 16 ? 1.f / 4 : 1.f;
    const float inv = amax > 0.f ? 127.f / amax : 0.f;
    if (!threadIdx.x) scale[blockIdx.x] = amax * norm / 127.f;
    int8_t* out_hi = hi + int64_t(blockIdx.x) * k;
    for (int base = warp * 256; base < k; base += 8 * 256) {
        float v[8];
        const bool valid = base + lane * 8 < k;
        load_act<T, ACT>(row, k, base + lane * 8, v, valid);
        if (ACT == ACT_RMS) apply_rms(v, inv_rms, act_weight, base + lane * 8, valid);
        rotate_chunk<GROUP>(v, lane);
        int q[8], r[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float t = v[j] * inv;
            q[j] = __float2int_rn(t);
            if (HILO) r[j] = __float2int_rn((t - float(q[j])) * 254.f);
        }
        if (valid) {
            *reinterpret_cast<uint2*>(out_hi + base + lane * 8) = make_uint2(pack_s8x4(q[0], q[1], q[2], q[3]), pack_s8x4(q[4], q[5], q[6], q[7]));
            if (HILO)
                *reinterpret_cast<uint2*>(lo + int64_t(blockIdx.x) * k + base + lane * 8) =
                    make_uint2(pack_s8x4(r[0], r[1], r[2], r[3]), pack_s8x4(r[4], r[5], r[6], r[7]));
        }
    }
}

// Single pass for K <= 2048 * CPW: each warp keeps its CPW rotated chunks in registers while the row maximum is reduced.
template<class T, int GROUP, bool HILO, int CPW, int ACT>
__global__ void __launch_bounds__(256) quantize_rows_registers(const T* __restrict__ x, int8_t* __restrict__ hi, int8_t* __restrict__ lo,
                                                               float* __restrict__ scale, int k, int64_t ldx, const T* __restrict__ act_weight, float eps) {
    __shared__ float warp_max[8], warp_ss[8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const T* row = x + int64_t(blockIdx.x) * ldx;
    float v[CPW][8];
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < CPW; ++i) {
        const int base = (warp + 8 * i) * 256;
        load_act<T, ACT>(row, k, base + lane * 8, v[i], base + lane * 8 < k);
    }
    if (ACT == ACT_RMS) {
        float ss = 0.f;
        #pragma unroll
        for (int i = 0; i < CPW; ++i)
            #pragma unroll
            for (int j = 0; j < 8; ++j) ss = fmaf(v[i][j], v[i][j], ss);
        const float inv_rms = rms_inverse(cta_sum(ss, warp_ss, lane, warp), k, eps);
        #pragma unroll
        for (int i = 0; i < CPW; ++i) {
            const int col = (warp + 8 * i) * 256 + lane * 8;
            apply_rms(v[i], inv_rms, act_weight, col, col < k);
        }
    }
    #pragma unroll
    for (int i = 0; i < CPW; ++i) {
        rotate_chunk<GROUP>(v[i], lane);
        #pragma unroll
        for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(v[i][j]));
    }
    #pragma unroll
    for (int d = 16; d; d >>= 1) amax = fmaxf(amax, shfl_xor(amax, d));
    if (!lane) warp_max[warp] = amax;
    __syncthreads();
    amax = warp_max[0];
    #pragma unroll
    for (int w = 1; w < 8; ++w) amax = fmaxf(amax, warp_max[w]);
    const float norm = GROUP == 256 ? 1.f / 16 : GROUP == 64 ? 1.f / 8 : GROUP == 16 ? 1.f / 4 : 1.f;
    const float inv = amax > 0.f ? 127.f / amax : 0.f;
    if (!threadIdx.x) scale[blockIdx.x] = amax * norm / 127.f;
    #pragma unroll
    for (int i = 0; i < CPW; ++i) {
        const int base = (warp + 8 * i) * 256 + lane * 8;
        if (base >= k) continue;
        int q[8], r[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float t = v[i][j] * inv;
            q[j] = __float2int_rn(t);
            if (HILO) r[j] = __float2int_rn((t - float(q[j])) * 254.f);
        }
        *reinterpret_cast<uint2*>(hi + int64_t(blockIdx.x) * k + base) = make_uint2(pack_s8x4(q[0], q[1], q[2], q[3]), pack_s8x4(q[4], q[5], q[6], q[7]));
        if (HILO)
            *reinterpret_cast<uint2*>(lo + int64_t(blockIdx.x) * k + base) = make_uint2(pack_s8x4(r[0], r[1], r[2], r[3]), pack_s8x4(r[4], r[5], r[6], r[7]));
    }
}

// Per-block activation scales: one FP32 scale per row and 128-element block of K, so that an outlier only coarsens its own
// block. Blocks are independent: a warp rotates a 256-element chunk (two blocks, 16 lanes each) and quantizes it, no row pass.
// The rotated 256-element chunk at `base` of `row` (lane l holds elements 8l..8l+7): two 128-element blocks of 16 lanes each.
template<int GROUP> DEVICE void store_chunk_blocks(float* v, int lane, int row, int base, int k, int8_t* __restrict__ q, float* __restrict__ scale) {
    const bool valid = base + lane * 8 < k;
    float amax = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(v[j]));
    #pragma unroll
    for (int d = 8; d; d >>= 1) amax = fmaxf(amax, shfl_xor(amax, d));  // within each half warp
    const float norm = GROUP == 256 ? 1.f / 16 : GROUP == 64 ? 1.f / 8 : GROUP == 16 ? 1.f / 4 : 1.f;
    const float inv = amax > 0.f ? 127.f / amax : 0.f;
    const int kblocks = (k + 127) / 128, block = base / 128 + (lane >> 4);
    if ((lane & 15) == 0 && block < kblocks) scale[int64_t(row) * kblocks + block] = amax * norm / 127.f;
    if (valid) {
        int r[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) r[j] = __float2int_rn(v[j] * inv);
        *reinterpret_cast<uint2*>(q + int64_t(row) * k + base + lane * 8) = make_uint2(pack_s8x4(r[0], r[1], r[2], r[3]), pack_s8x4(r[4], r[5], r[6], r[7]));
    }
}

// A warp per 256-element chunk, no row pass; RMSNorm takes the rows' 1 / RMS from row_inv_rms.
template<class T, int GROUP, int ACT>
__global__ void __launch_bounds__(256) quantize_blocks(const T* __restrict__ x, int8_t* __restrict__ q, float* __restrict__ scale, int k, int64_t ldx,
                                                       const T* __restrict__ act_weight, const float* __restrict__ inv_rms) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, base = (blockIdx.x * 8 + warp) * 256;
    if (base >= k) return;
    const int row = blockIdx.y;
    float v[8];
    load_act<T, ACT>(x + int64_t(row) * ldx, k, base + lane * 8, v, base + lane * 8 < k);
    if (ACT == ACT_RMS) apply_rms(v, inv_rms[row], act_weight, base + lane * 8, base + lane * 8 < k);
    rotate_chunk<GROUP>(v, lane);
    store_chunk_blocks<GROUP>(v, lane, row, base, k, q, scale);
}

// RMSNorm rows of K <= 2048 * CPW: one CTA per row keeps its chunks in registers while the sum of squares is reduced (one read).
template<class T, int GROUP, int CPW>
__global__ void __launch_bounds__(256) quantize_blocks_rms(const T* __restrict__ x, int8_t* __restrict__ q, float* __restrict__ scale, int k, int64_t ldx,
                                                           const T* __restrict__ act_weight, float eps) {
    __shared__ float warp_ss[8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, row = blockIdx.x;
    float v[CPW][8], ss = 0.f;
    #pragma unroll
    for (int i = 0; i < CPW; ++i) {
        const int col = (warp + 8 * i) * 256 + lane * 8;
        load8(x + int64_t(row) * ldx + col, v[i], col < k);
        #pragma unroll
        for (int j = 0; j < 8; ++j) ss = fmaf(v[i][j], v[i][j], ss);
    }
    const float inv_rms = rms_inverse(cta_sum(ss, warp_ss, lane, warp), k, eps);
    #pragma unroll
    for (int i = 0; i < CPW; ++i) {
        const int base = (warp + 8 * i) * 256;
        if (base >= k) break;  // warp-uniform
        apply_rms(v[i], inv_rms, act_weight, base + lane * 8, base + lane * 8 < k);
        rotate_chunk<GROUP>(v[i], lane);
        store_chunk_blocks<GROUP>(v[i], lane, row, base, k, q, scale);
    }
}

// Few-token paths: the input operation and ConvRot rotation applied in FP32 and rounded once to T (out: M x K contiguous).
template<class T, int GROUP, int ACT>
__global__ void __launch_bounds__(256) transform_rows(const T* __restrict__ x, T* __restrict__ out, int k, int64_t ldx,
                                                      const T* __restrict__ act_weight, float eps) {
    __shared__ float warp_ss[8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const T* row = x + int64_t(blockIdx.x) * ldx;
    float inv_rms = 1.f;
    if (ACT == ACT_RMS) {
        float ss = 0.f;
        for (int base = warp * 256; base < k; base += 8 * 256) {
            float v[8];
            load8(row + base + lane * 8, v, base + lane * 8 < k);
            #pragma unroll
            for (int j = 0; j < 8; ++j) ss = fmaf(v[j], v[j], ss);
        }
        inv_rms = rms_inverse(cta_sum(ss, warp_ss, lane, warp), k, eps);
    }
    constexpr float norm = GROUP == 256 ? 1.f / 16 : GROUP == 64 ? 1.f / 8 : GROUP == 16 ? 1.f / 4 : 1.f;
    for (int base = warp * 256; base < k; base += 8 * 256) {
        float v[8];
        const bool valid = base + lane * 8 < k;
        load_act<T, ACT>(row, k, base + lane * 8, v, valid);
        if (ACT == ACT_RMS) apply_rms(v, inv_rms, act_weight, base + lane * 8, valid);
        rotate_chunk<GROUP>(v, lane);
        if (valid) {
            uint4 packed;
            uint32_t* p = reinterpret_cast<uint32_t*>(&packed);
            #pragma unroll
            for (int j = 0; j < 4; ++j) p[j] = Pack2<T>::make(v[2 * j] * norm, v[2 * j + 1] * norm);
            *reinterpret_cast<uint4*>(out + int64_t(blockIdx.x) * k + base + lane * 8) = packed;
        }
    }
}

#if !WGP_AMD
// ---------------------------------------------------------------------------------------------------------------------
// INT8 x INT8 -> INT32 tensor-core GEMM, multistage cp.async pipeline, XOR-swizzled shared memory.
// A: (M, K) row-major INT8 activations with per-row scales. B: (N, K) row-major INT8 weights with per-column scales.
// With HILO, A has a second residual operand sharing the row scales: acc_hi + acc_lo / 254.
//
// Persistent CTAs process whole tiles in waves. The tiles of an incomplete last wave are split along K over the CTAs
// (stream-K): each piece stores its exact INT32 partial sums in its own workspace slot and the last piece to arrive adds
// the others and applies the epilogue. Integer sums make the result independent of arrival order; no CTA ever waits.
// ---------------------------------------------------------------------------------------------------------------------
template<int BM, int BN, int BK, int WM, int WN, int STAGES, bool HILO>
struct GemmConfig {
    static constexpr int WARPS_M = BM / WM, WARPS_N = BN / WN, THREADS = 32 * WARPS_M * WARPS_N;
    static constexpr int MI = WM / 16, NI = WN / 8, CHUNKS = BK / 16, KSTEPS = BK / 32;
    static constexpr int A_BYTES = BM * BK, B_BYTES = BN * BK, STAGE_BYTES = A_BYTES * (HILO ? 2 : 1) + B_BYTES;
    static constexpr int SMEM = STAGES * STAGE_BYTES;
    static constexpr int ROW_STEP = THREADS / CHUNKS, A_PER = BM / ROW_STEP, B_PER = BN / ROW_STEP;
    static constexpr int ACC = HILO ? 2 : 1;
};

struct Schedule {
    int tiles_m, tiles_n, k_tiles;
    int dp_tiles;          // leading tiles computed whole, tile t by CTA t % gridDim.x
    int sk_tiles;          // trailing tiles split along K over the first sk_ctas CTAs
    int sk_ctas;
    long long sk_units;    // sk_tiles * k_tiles
    int grid;
};

// 16-byte chunk c of row r: swizzle so that the 8 rows read by one ldmatrix phase hit distinct bank groups.
template<int CHUNKS> DEVICE int swizzle(int r, int c) {
    if (CHUNKS == 4) return r * 64 + ((c ^ ((r >> 1) & 3)) << 4);
    return r * (CHUNKS * 16) + ((c ^ (r & 7)) << 4);
}

DEVICE int sk_cta_of(long long unit, const Schedule& s) { return int(((unit + 1) * s.sk_ctas - 1) / s.sk_units); }

template<class T, int BM, int BN, int BK, int WM, int WN, int STAGES, bool HILO, int CTAS, bool BLOCKS>
__global__ void __launch_bounds__(GemmConfig<BM, BN, BK, WM, WN, STAGES, HILO>::THREADS, CTAS)
gemm_s8(const int8_t* __restrict__ a, const int8_t* __restrict__ a_lo, const float* __restrict__ sa,
        const int8_t* __restrict__ b, const float* __restrict__ sb, const T* __restrict__ bias, const T* residual, const T* __restrict__ rscale,
        T* y, int m, int n, int k, Schedule sched, int* __restrict__ workspace, int* __restrict__ counters) {
    using C = GemmConfig<BM, BN, BK, WM, WN, STAGES, HILO>;
    constexpr int MI = C::MI, NI = C::NI;
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ int s_last;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int warp_m = warp / C::WARPS_N, warp_n = warp % C::WARPS_N;
    const uint32_t base = smem_u32(smem);
    const int chunk = tid % C::CHUNKS, row0 = tid / C::CHUNKS;
    const int g = lane >> 2, t4 = lane & 3;

    int acc[C::ACC][MI][NI][4];
    // BLOCKS: sa holds one scale per row and 128-element K block. Each block's exact INT32 sums are scaled into FP32 accumulators.
    constexpr int KPB = 128 / BK;  // k-tiles per scale block
    float accf[BLOCKS ? MI : 1][BLOCKS ? NI : 1][4];
    // The scales of a block's BM rows come with the stage of its first k-tile (cp.async: no registers held while in flight) into a ring
    // of three blocks; the slot of block b is refilled for block b + 3, whose stage is issued after every warp folded block b.
    __shared__ float block_scales[BLOCKS ? 3 : 1][BLOCKS ? BM : 1];
    const int kblocks = (k + 127) / 128;
    int work_m0 = 0, work_kb = 0;

    auto tile_origin = [&](int t, int& m0, int& n0) {
        // Grouped rasterization keeps a band of A rows and the B columns it meets resident in L2.
        constexpr int GROUP_M = 8;
        const int width = GROUP_M * sched.tiles_n, first = (t / width) * GROUP_M, rows = min(sched.tiles_m - first, GROUP_M);
        m0 = (first + (t % width) % rows) * BM;
        n0 = ((t % width) / rows) * BN;
    };

    // One unit of work: k-tiles [kb, ke) of the tile at (m0, n0); j >= 0 marks a split (stream-K) tile.
    struct Work { int m0, n0, kb, ke, j; };
    int dp_tile = blockIdx.x;
    long long u = 0, u_end = 0;
    if (int(blockIdx.x) < sched.sk_ctas) {
        u = blockIdx.x * sched.sk_units / sched.sk_ctas;
        u_end = (blockIdx.x + 1) * sched.sk_units / sched.sk_ctas;
    }
    auto next_work = [&](Work& w) {
        if (dp_tile < sched.dp_tiles) {
            tile_origin(dp_tile, w.m0, w.n0);
            w.kb = 0; w.ke = sched.k_tiles; w.j = -1;
            dp_tile += gridDim.x;
            return true;
        }
        if (u >= u_end) return false;
        w.j = int(u / sched.k_tiles); w.kb = int(u % sched.k_tiles);
        const long long span = w.kb + (u_end - u);
        w.ke = span < sched.k_tiles ? int(span) : sched.k_tiles;
        tile_origin(sched.dp_tiles + w.j, w.m0, w.n0);
        u += w.ke - w.kb;
        return true;
    };

    // Per-thread copy sources of the current tile: fixed chunk column, rows strided by ROW_STEP.
    const int8_t* a_src[C::A_PER]; const int8_t* b_src[C::B_PER];
    bool a_ok[C::A_PER], b_ok[C::B_PER];
    auto setup = [&](const Work& w) {
        work_m0 = w.m0;
        work_kb = w.kb;
        #pragma unroll
        for (int i = 0; i < C::A_PER; ++i) {
            const int r = w.m0 + row0 + i * C::ROW_STEP;
            a_ok[i] = r < m;
            a_src[i] = a + (a_ok[i] ? int64_t(r) * k + chunk * 16 : 0);
        }
        #pragma unroll
        for (int i = 0; i < C::B_PER; ++i) {
            const int r = w.n0 + row0 + i * C::ROW_STEP;
            b_ok[i] = r < n;
            b_src[i] = b + (b_ok[i] ? int64_t(r) * k + chunk * 16 : 0);
        }
    };
    // BLOCKS: the activation scales of k-tile kt's block, with the tile that starts the block (or the work)
    auto load_scales = [&](int kt) {
        if (BLOCKS && (kt % KPB == 0 || kt == work_kb) && tid < BM) {
            const int row = work_m0 + tid;
            cp_async4(smem_u32(&block_scales[(kt / KPB) % 3][tid]), row < m ? sa + int64_t(row) * kblocks + kt / KPB : sa, row < m);
        }
    };
    auto load_stage = [&](int slot, int kt) {
        const int k0 = kt * BK;
        const bool k_ok = k0 + chunk * 16 < k;
        const uint32_t sa_base = base + slot * C::STAGE_BYTES, sb_base = sa_base + C::A_BYTES * (HILO ? 2 : 1);
        #pragma unroll
        for (int i = 0; i < C::A_PER; ++i) {
            const bool ok = a_ok[i] && k_ok;
            const uint32_t dst = sa_base + swizzle<C::CHUNKS>(row0 + i * C::ROW_STEP, chunk);
            cp_async16(dst, ok ? a_src[i] + k0 : a, ok);
            if (HILO) cp_async16(dst + C::A_BYTES, ok ? a_lo + (a_src[i] - a) + k0 : a_lo, ok);
        }
        #pragma unroll
        for (int i = 0; i < C::B_PER; ++i) {
            const bool ok = b_ok[i] && k_ok;
            cp_async16(sb_base + swizzle<C::CHUNKS>(row0 + i * C::ROW_STEP, chunk), ok ? b_src[i] + k0 : b, ok);
        }
        load_scales(kt);
    };
#if WGP_SM75
    // Turing: the k-tile written at the end of tile kt (kt + STAGES - 1) is read from global memory into registers at the end of tile
    // kt - 1, so that its load latency is hidden by a whole tile of MMAs (synchronous copies would stall every warp at once).
    uint4 staged_a[C::A_PER][HILO ? 2 : 1], staged_b[C::B_PER];
    auto fetch_stage = [&](int kt) {
        const int k0 = kt * BK;
        const bool k_ok = k0 + chunk * 16 < k;
        #pragma unroll
        for (int i = 0; i < C::A_PER; ++i) {
            const bool ok = a_ok[i] && k_ok;
            staged_a[i][0] = ok ? __ldcg(reinterpret_cast<const uint4*>(a_src[i] + k0)) : make_uint4(0, 0, 0, 0);
            if (HILO) staged_a[i][HILO ? 1 : 0] = ok ? __ldcg(reinterpret_cast<const uint4*>(a_lo + (a_src[i] - a) + k0)) : make_uint4(0, 0, 0, 0);
        }
        #pragma unroll
        for (int i = 0; i < C::B_PER; ++i)
            staged_b[i] = b_ok[i] && k_ok ? __ldcg(reinterpret_cast<const uint4*>(b_src[i] + k0)) : make_uint4(0, 0, 0, 0);
    };
    auto store_stage = [&](int slot) {
        const uint32_t sa_base = base + slot * C::STAGE_BYTES, sb_base = sa_base + C::A_BYTES * (HILO ? 2 : 1);
        #pragma unroll
        for (int i = 0; i < C::A_PER; ++i) {
            const uint32_t dst = sa_base + swizzle<C::CHUNKS>(row0 + i * C::ROW_STEP, chunk);
            #pragma unroll
            for (int h = 0; h < (HILO ? 2 : 1); ++h)
                asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(dst + h * C::A_BYTES), "r"(staged_a[i][h].x), "r"(staged_a[i][h].y),
                             "r"(staged_a[i][h].z), "r"(staged_a[i][h].w));
        }
        #pragma unroll
        for (int i = 0; i < C::B_PER; ++i)
            asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(sb_base + swizzle<C::CHUNKS>(row0 + i * C::ROW_STEP, chunk)),
                         "r"(staged_b[i].x), "r"(staged_b[i].y), "r"(staged_b[i].z), "r"(staged_b[i].w));
    };
#endif
    auto prologue = [&](const Work& w) {
        #pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) {
            if (w.kb + s < w.ke) load_stage(s, w.kb + s);
            cp_async_commit();
        }
#if WGP_SM75
        if (w.kb + STAGES - 1 < w.ke) fetch_stage(w.kb + STAGES - 1);
#endif
    };

    uint32_t af[2][C::ACC][MI][4], bf[2][NI][2];
    // ldmatrix offsets: the swizzle's XOR term is the same for rows 16 apart, so the fragments of a k-step are one per-thread offset
    // plus constants (16 rows = 16 * CHUNKS * 16 bytes).
    uint32_t a_frag[C::KSTEPS], b_frag[C::KSTEPS];
    #pragma unroll
    for (int ks = 0; ks < C::KSTEPS; ++ks) {
        a_frag[ks] = swizzle<C::CHUNKS>(warp_m * WM + (lane & 15), ks * 2 + (lane >> 4));
        b_frag[ks] = swizzle<C::CHUNKS>(warp_n * WN + (lane & 7) + ((lane >> 4) << 3), ks * 2 + ((lane >> 3) & 1));
    }
    constexpr int ROWS16 = 16 * C::CHUNKS * 16;
    auto load_frags = [&](int buf, int slot, int ks) {
        const uint32_t sa_base = base + slot * C::STAGE_BYTES + a_frag[ks], sb_base = base + slot * C::STAGE_BYTES + C::A_BYTES * (HILO ? 2 : 1) + b_frag[ks];
        #pragma unroll
        for (int mi = 0; mi < MI; ++mi)
            #pragma unroll
            for (int h = 0; h < C::ACC; ++h) ldmatrix_x4(af[buf][h][mi], sa_base + h * C::A_BYTES + mi * ROWS16);
        #pragma unroll
        for (int ni = 0; ni < NI; ni += 2) {
            uint32_t t[4];
            ldmatrix_x4(t, sb_base + (ni / 2) * ROWS16);
            bf[buf][ni][0] = t[0]; bf[buf][ni][1] = t[1]; bf[buf][ni + 1][0] = t[2]; bf[buf][ni + 1][1] = t[3];
        }
    };

    // Accumulate the k-tiles of w, whose prologue was issued.
    auto mainloop = [&](const Work& w) {
        #pragma unroll
        for (int h = 0; h < C::ACC; ++h)
            #pragma unroll
            for (int mi = 0; mi < MI; ++mi)
                #pragma unroll
                for (int ni = 0; ni < NI; ++ni)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) acc[h][mi][ni][j] = BLOCKS ? BLOCK_BIAS : 0;
        if (BLOCKS) {
            #pragma unroll
            for (int mi = 0; mi < MI; ++mi)
                #pragma unroll
                for (int ni = 0; ni < NI; ++ni)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) accf[mi][ni][j] = 0.f;
        }
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        load_frags(0, 0, 0);
        int slot = 0;
        for (int kt = w.kb; kt < w.ke;) {
            // BLOCKS: the k-tiles of one scale block, then its fold, once and unpredicated (a fold inside the k-tile loop is issued,
            // predicated off, on every k-tile).
            const int block_end = BLOCKS ? min(w.ke, (kt / KPB + 1) * KPB) : w.ke;
            for (; kt < block_end; ++kt) {
                #pragma unroll
                for (int ks = 0; ks < C::KSTEPS; ++ks) {
                    const int buf = ks & 1;
                    if (ks == C::KSTEPS - 1) {
                        // Tile kt + 1 must have landed and every warp must be done with tile kt - 1, whose slot is refilled.
                        cp_async_wait<STAGES - 3>();
                        __syncthreads();
                        const int next = kt + STAGES - 1;
#if WGP_SM75
                        if (next < w.ke) {
                            store_stage((slot + STAGES - 1) % STAGES);
                            load_scales(next);
                            if (STAGES == 2) __syncthreads();  // with two stages the tile just stored is the next one read
                            if (next + 1 < w.ke) fetch_stage(next + 1);
                        }
#else
                        if (next < w.ke) load_stage((slot + STAGES - 1) % STAGES, next);
#endif
                        cp_async_commit();
                        if (kt + 1 < w.ke) load_frags(buf ^ 1, (slot + 1) % STAGES, 0);
                    } else {
                        load_frags(buf ^ 1, slot, ks + 1);
                    }
                    #pragma unroll
                    for (int mi = 0; mi < MI; ++mi)
                        #pragma unroll
                        for (int ni = 0; ni < NI; ++ni)
                            #pragma unroll
                            for (int h = 0; h < C::ACC; ++h) mma_s8(acc[h][mi][ni], af[buf][h][mi], bf[buf][ni]);
                }
                slot = (slot + 1) % STAGES;
            }
            if (BLOCKS) {
                const float* scales = block_scales[((kt - 1) / KPB) % 3] + warp_m * WM + g;
                #pragma unroll
                for (int mi = 0; mi < MI; ++mi) {
                    const float s0 = scales[mi * 16], s1 = scales[mi * 16 + 8];
                    #pragma unroll
                    for (int ni = 0; ni < NI; ++ni) {
                        fold_block_pair(&accf[mi][ni][0], &acc[0][mi][ni][0], s0);
                        fold_block_pair(&accf[mi][ni][2], &acc[0][mi][ni][2], s1);
                    }
                }
            }
        }
        __syncthreads();  // every warp is done with shared memory before the next prologue refills it
    };

    auto epilogue = [&](int m0, int n0) {
        float2 col_scale[NI], col_bias[NI];
        #pragma unroll
        for (int ni = 0; ni < NI; ++ni) {
            const int col = n0 + warp_n * WN + ni * 8 + t4 * 2;
            col_scale[ni] = col + 1 < n ? *reinterpret_cast<const float2*>(sb + col) : make_float2(col < n ? sb[col] : 0.f, 0.f);
            col_bias[ni] = make_float2(bias && col < n ? to_float(bias[col]) : 0.f, bias && col + 1 < n ? to_float(bias[col + 1]) : 0.f);
        }
        float row_scale[MI][2];
        #pragma unroll
        for (int mi = 0; mi < MI; ++mi)
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int row = m0 + warp_m * WM + mi * 16 + g + half * 8;
                row_scale[mi][half] = BLOCKS ? 1.f : row < m ? sa[row] : 0.f;
            }
        auto value = [&](int mi, int ni, int half, int e) {
            float v = BLOCKS ? accf[mi < MI ? mi : 0][BLOCKS ? ni : 0][half * 2 + e] : float(acc[0][mi][ni][half * 2 + e]);
            if (HILO) v = fmaf(float(acc[C::ACC - 1][mi][ni][half * 2 + e]), 1.f / 254.f, v);
            return fmaf(v * row_scale[mi][half], e ? col_scale[ni].y : col_scale[ni].x, e ? col_bias[ni].y : col_bias[ni].x);
        };
        // Both outputs of a fragment pair, with the residual (read before the tile is written: y may be the residual itself).
        auto pair = [&](int mi, int ni, int half) {
            float2 v = make_float2(value(mi, ni, half, 0), value(mi, ni, half, 1));
            if (residual) {
                const int row = m0 + warp_m * WM + mi * 16 + g + half * 8, col = n0 + warp_n * WN + ni * 8 + t4 * 2;
                if (row < m && col < n) {
                    const T* r = residual + int64_t(row) * n + col;
                    float2 rv;
                    if (col + 1 < n && (n & 1) == 0) {
                        const uint32_t raw = *reinterpret_cast<const uint32_t*>(r);
                        rv = make_float2(to_float(reinterpret_cast<const T*>(&raw)[0]), to_float(reinterpret_cast<const T*>(&raw)[1]));
                    } else {
                        rv = make_float2(to_float(r[0]), col + 1 < n ? to_float(r[1]) : 0.f);
                    }
                    // residual scales read here, not held per column: the widest tile has no registers to spare
                    const float rs0 = rscale ? to_float(rscale[col]) : 1.f, rs1 = rscale && col + 1 < n ? to_float(rscale[col + 1]) : 1.f;
                    v.x = fmaf(v.x, rs0, rv.x);
                    v.y = fmaf(v.y, rs1, rv.y);
                }
            }
            return v;
        };
        // Turing's 2-stage 128x256 tile has no room for the staged output tile: its rows are stored directly.
        constexpr bool STAGED_OUTPUT = BM * (BN + 8) * sizeof(T) <= C::SMEM;
        if constexpr (STAGED_OUTPUT) if ((n & 7) == 0) {
            // Like CUTLASS's epilogue: stage the tile in shared memory (free after the mainloop; rows padded by 16 bytes so that
            // the 8 rows of a fragment hit distinct banks), then store whole rows with coalesced 16-byte accesses.
            constexpr int LD = BN + 8, VECS = BN / 8;
            static_assert(BM * LD * sizeof(T) <= C::SMEM, "output tile must fit in the pipeline's shared memory");
            T* tile = reinterpret_cast<T*>(smem);
            #pragma unroll
            for (int mi = 0; mi < MI; ++mi)
                #pragma unroll
                for (int half = 0; half < 2; ++half)
                    #pragma unroll
                    for (int ni = 0; ni < NI; ++ni) {
                        const float2 v = pair(mi, ni, half);
                        *reinterpret_cast<uint32_t*>(tile + (warp_m * WM + mi * 16 + g + half * 8) * LD + warp_n * WN + ni * 8 + t4 * 2) = Pack2<T>::make(v.x, v.y);
                    }
            __syncthreads();
            #pragma unroll 4
            for (int i = tid; i < BM * VECS; i += C::THREADS) {
                const int r = i / VECS, c = (i % VECS) * 8, row = m0 + r, col = n0 + c;
                if (row < m && col < n) *reinterpret_cast<uint4*>(y + int64_t(row) * n + col) = *reinterpret_cast<const uint4*>(tile + r * LD + c);
            }
            __syncthreads();  // the next prologue refills the shared memory
            return;
        }
        #pragma unroll
        for (int mi = 0; mi < MI; ++mi) {
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int row = m0 + warp_m * WM + mi * 16 + g + half * 8;
                if (row >= m) continue;
                T* out = y + int64_t(row) * n;
                #pragma unroll
                for (int ni = 0; ni < NI; ++ni) {
                    const int col = n0 + warp_n * WN + ni * 8 + t4 * 2;
                    const float2 v = pair(mi, ni, half);
                    if (col + 1 < n && (n & 1) == 0) *reinterpret_cast<uint32_t*>(out + col) = Pack2<T>::make(v.x, v.y);
                    else if (col < n) { out[col] = T(v.x); if (col + 1 < n) out[col + 1] = T(v.y); }
                }
            }
        }
    };

    // Split tiles: each piece stores its exact INT32 partial sums in its own slot; the last to arrive adds the others.
    auto finish = [&](const Work& w) {
        if (w.j >= 0) {
            const int first = sk_cta_of((long long)w.j * sched.k_tiles, sched), last = sk_cta_of((long long)(w.j + 1) * sched.k_tiles - 1, sched);
            if (first != last) {
                // Slot blockIdx.x + j is unique: CTA ranges are increasing, so no other CTA/tile pair has the same sum.
                constexpr int VECS = C::ACC * MI * NI;
                int4* slots = reinterpret_cast<int4*>(workspace);
                int4* own = slots + int64_t(blockIdx.x + w.j) * VECS * C::THREADS + tid;
                #pragma unroll
                for (int h = 0; h < C::ACC; ++h)
                    #pragma unroll
                    for (int mi = 0; mi < MI; ++mi)
                        #pragma unroll
                        for (int ni = 0; ni < NI; ++ni)
                            __stcg(own + ((h * MI + mi) * NI + ni) * C::THREADS,
                                   BLOCKS ? *reinterpret_cast<const int4*>(accf[mi][ni]) : *reinterpret_cast<const int4*>(acc[h][mi][ni]));
                __threadfence();
                __syncthreads();
                if (!tid) s_last = atomicAdd(counters + w.j, 1) == last - first;
                __syncthreads();
                if (!s_last) return;
                __threadfence();
                if (BLOCKS) {
                    // Floating-point sums: add the pieces in CTA order, this one included, whichever arrives last.
                    float sum[BLOCKS ? MI : 1][BLOCKS ? NI : 1][4] = {};
                    for (int c = first; c <= last; ++c) {
                        const int4* piece = slots + int64_t(c + w.j) * VECS * C::THREADS + tid;
                        #pragma unroll
                        for (int mi = 0; mi < MI; ++mi)
                            #pragma unroll
                            for (int ni = 0; ni < NI; ++ni) {
                                float4 v;
                                if (c == int(blockIdx.x)) v = *reinterpret_cast<const float4*>(accf[mi][ni]);
                                else { const int4 raw = __ldcg(piece + (mi * NI + ni) * C::THREADS); v = *reinterpret_cast<const float4*>(&raw); }
                                sum[mi][ni][0] += v.x; sum[mi][ni][1] += v.y; sum[mi][ni][2] += v.z; sum[mi][ni][3] += v.w;
                            }
                    }
                    #pragma unroll
                    for (int mi = 0; mi < MI; ++mi)
                        #pragma unroll
                        for (int ni = 0; ni < NI; ++ni)
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) accf[mi][ni][j] = sum[mi][ni][j];
                }
                for (int c = first; c <= last && !BLOCKS; ++c) {
                    if (c == int(blockIdx.x)) continue;
                    const int4* other = slots + int64_t(c + w.j) * VECS * C::THREADS + tid;
                    #pragma unroll
                    for (int h = 0; h < C::ACC; ++h)
                        #pragma unroll
                        for (int mi = 0; mi < MI; ++mi)
                            #pragma unroll
                            for (int ni = 0; ni < NI; ++ni) {
                                const int4 v = __ldcg(other + ((h * MI + mi) * NI + ni) * C::THREADS);
                                acc[h][mi][ni][0] += v.x; acc[h][mi][ni][1] += v.y; acc[h][mi][ni][2] += v.z; acc[h][mi][ni][3] += v.w;
                            }
                }
                if (!tid) counters[w.j] = 0;
            }
        }
        epilogue(w.m0, w.n0);
    };

    Work w;
    if (!next_work(w)) return;
    setup(w);
    prologue(w);
    for (;;) {
        mainloop(w);
        finish(w);
        if (!next_work(w)) break;
        setup(w);
        prologue(w);
    }
}


// ---------------------------------------------------------------------------------------------------------------------
// Few tokens (autoregressive decoding): weight-only INT8. The weights are streamed once from memory at full bandwidth and
// the activations are not quantized: INT8 weights convert exactly to BF16/FP16 in registers and products accumulate in
// FP32 on tensor cores ("swap AB": 16 output channels per MMA row block, up to 8 tokens per MMA column block).
// Each thread loads 16 contiguous weight bytes of a row; the K order inside a 64-element chunk is permuted identically
// for weights and activations, which turns those bytes into valid MMA fragments without shared memory.
// ---------------------------------------------------------------------------------------------------------------------
template<class T> DEVICE void mma_16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
#if WGP_SM75
    // FP16 only (sm_75 has no BF16 MMA; BF16 few-token calls take the lines GEMV): two m16n8k8 on the k 0-7 / 8-15 fragments
    if constexpr (std::is_same<T, __nv_bfloat16>::value)
        __trap();
    else
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                     "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%7,%8}, {%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(b0), "r"(a[2]), "r"(a[3]), "r"(b1));
#else
    if constexpr (std::is_same<T, __nv_bfloat16>::value)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    else
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

// Two signed INT8 bytes (selected by SEL from w) as an exact BF16/FP16 pair: (base | low7) - (base | sign bit), where
// base is 128.0 in BF16 (0x4300) or 1024.0 in FP16 (0x6400) so that the 7 low bits are integer mantissa steps.
template<class T, uint32_t SEL> DEVICE uint32_t int8_pair(uint32_t low7, uint32_t sign) {
    constexpr uint32_t base = std::is_same<T, __nv_bfloat16>::value ? 0x43434343u : 0x64646464u;
    uint32_t v, o;
    asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(v) : "r"(low7), "r"(base), "n"(SEL));
    asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(o) : "r"(sign), "r"(base), "n"(SEL));
    if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        __nv_bfloat162 r = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&v), *reinterpret_cast<__nv_bfloat162*>(&o));
        return *reinterpret_cast<uint32_t*>(&r);
    } else {
        __half2 r = __hsub2(*reinterpret_cast<__half2*>(&v), *reinterpret_cast<__half2*>(&o));
        return *reinterpret_cast<uint32_t*>(&r);
    }
}

DEVICE uint4 ldg_stream(const void* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}

// CTA: 16 output channels x a range of K chunks (64 * V elements), 8 warps interleaving chunks. MT blocks of 8 tokens.
// Thread t of a quad reads 16 * V contiguous bytes of its two weight rows per chunk (longer runs suit DRAM bursts).
template<class T, int MT, int UNROLL, int V>
__global__ void __launch_bounds__(256) w8a16_small(const T* __restrict__ x, int64_t ldx, const int8_t* __restrict__ w, const float* __restrict__ sw,
                                                   const T* __restrict__ bias, const T* residual, const T* __restrict__ rscale, T* y,
                                                   int m, int n, int k, float* __restrict__ slots, int* __restrict__ counters) {
    constexpr int CHUNK = 64 * V;
    __shared__ float partial[8][MT * 8][17];
    __shared__ int s_last;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int ch0 = blockIdx.x * 16 + g, ch1 = ch0 + 8;
    const int chunks = k / CHUNK, splits = gridDim.y;
    const int c_begin = int((long long)blockIdx.y * chunks / splits), c_end = int((long long)(blockIdx.y + 1) * chunks / splits);
    const bool ok0 = ch0 < n, ok1 = ch1 < n;
    const int8_t* w0 = w + int64_t(ok0 ? ch0 : 0) * k + 16 * V * t;
    const int8_t* w1 = w + int64_t(ok1 ? ch1 : 0) * k + 16 * V * t;
    const T* xr[MT];
    bool xok[MT];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int row = mt * 8 + g;
        xok[mt] = row < m;
        xr[mt] = x + (xok[mt] ? int64_t(row) * ldx : 0) + 16 * V * t;
    }
    float acc[MT][4] = {};
    for (int c = c_begin + warp; c < c_end; c += 8 * UNROLL) {
        uint4 wa[UNROLL][V], wb[UNROLL][V];
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int cc = c + 8 * u;
            const bool in = cc < c_end;
            #pragma unroll
            for (int v = 0; v < V; ++v) {
                wa[u][v] = in && ok0 ? ldg_stream(w0 + cc * CHUNK + 16 * v) : make_uint4(0, 0, 0, 0);
                wb[u][v] = in && ok1 ? ldg_stream(w1 + cc * CHUNK + 16 * v) : make_uint4(0, 0, 0, 0);
            }
        }
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            // No early exit: chunks past the range have zero weights and activations (an exit would stop load hoisting).
            const int cc = c + 8 * u;
            const bool in = cc < c_end;
            #pragma unroll
            for (int v = 0; v < V; ++v) {
                uint4 xv[MT][2];
                #pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    const T* p = xr[mt] + cc * CHUNK + 16 * v;
                    xv[mt][0] = xok[mt] && in ? *reinterpret_cast<const uint4*>(p) : make_uint4(0, 0, 0, 0);
                    xv[mt][1] = xok[mt] && in ? *reinterpret_cast<const uint4*>(p + 8) : make_uint4(0, 0, 0, 0);
                }
                const uint32_t* ra = reinterpret_cast<const uint32_t*>(&wa[u][v]);
                const uint32_t* rb = reinterpret_cast<const uint32_t*>(&wb[u][v]);
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    // Logical k (2t, 2t+1 | 2t+8, 2t+9) of this MMA = physical elements 16 V t + 16 v + 4q + (0, 1 | 2, 3).
                    const uint32_t la = ra[q] & 0x7f7f7f7fu, sa_ = ra[q] & 0x80808080u, lb = rb[q] & 0x7f7f7f7fu, sb_ = rb[q] & 0x80808080u;
                    uint32_t af[4];
                    af[0] = int8_pair<T, 0x5140>(la, sa_);
                    af[1] = int8_pair<T, 0x5140>(lb, sb_);
                    af[2] = int8_pair<T, 0x5342>(la, sa_);
                    af[3] = int8_pair<T, 0x5342>(lb, sb_);
                    #pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        const uint32_t* xs = reinterpret_cast<const uint32_t*>(&xv[mt][q >> 1]);
                        mma_16816<T>(acc[mt], af, xs[(q & 1) * 2], xs[(q & 1) * 2 + 1]);
                    }
                }
            }
        }
    }
    // acc[mt]: (channel g, tokens 2t, 2t+1) and (channel g + 8, same tokens) of token block mt. Fixed-order reductions.
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        partial[warp][mt * 8 + 2 * t][g] = acc[mt][0];
        partial[warp][mt * 8 + 2 * t + 1][g] = acc[mt][1];
        partial[warp][mt * 8 + 2 * t][g + 8] = acc[mt][2];
        partial[warp][mt * 8 + 2 * t + 1][g + 8] = acc[mt][3];
    }
    __syncthreads();
    const int outputs = MT * 8 * 16;
    float* own = slots ? slots + (int64_t(blockIdx.x) * splits + blockIdx.y) * outputs : nullptr;
    for (int i = threadIdx.x; i < outputs; i += 256) {
        const int token = i / 16, c = i % 16;
        float s = 0.f;
        #pragma unroll
        for (int wi = 0; wi < 8; ++wi) s += partial[wi][token][c];
        partial[0][token][c] = s;
        if (own) __stcg(own + i, s);
    }
    if (splits > 1) {
        __threadfence();
        __syncthreads();
        if (!threadIdx.x) s_last = atomicAdd(counters + blockIdx.x, 1) == splits - 1;
        __syncthreads();
        if (!s_last) return;
        __threadfence();
        const float* tile = slots + int64_t(blockIdx.x) * splits * outputs;
        for (int i = threadIdx.x; i < outputs; i += 256) {
            float s = 0.f;
            for (int si = 0; si < splits; ++si) s += __ldcg(tile + si * outputs + i);
            partial[0][i / 16][i % 16] = s;
        }
        if (!threadIdx.x) counters[blockIdx.x] = 0;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < outputs; i += 256) {
        const int token = i / 16, c = blockIdx.x * 16 + i % 16;
        if (token < m && c < n) {
            float v = partial[0][token][i % 16] * sw[c] + (bias ? to_float(bias[c]) : 0.f);
            if (residual) v = fmaf(v, rscale ? to_float(rscale[c]) : 1.f, to_float(residual[int64_t(token) * n + c]));
            y[int64_t(token) * n + c] = T(v);
        }
    }
}
#endif  // !WGP_AMD

#if WGP_AMD
// ---------------------------------------------------------------------------------------------------------------------
// AMD RDNA4 (gfx12, wave32) INT8 GEMM on rocWMMA 16x16x16 INT8 x INT8 -> INT32 fragments: the operands, modes (row scales, HILO
// residual operand, per-block scales) and epilogue (bias, residual, one rounding) of gemm_s8, as whole BM x 128 tiles (no stream-K),
// operands staged through registers into double-buffered LDS. The row and column of every accumulator element are read from
// rocWMMA itself (an accumulator fragment loaded from a tile of indices), not assumed from the instruction layout.
// ---------------------------------------------------------------------------------------------------------------------
using FragA = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, int8_t, rocwmma::row_major>;
using FragB = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, int8_t, rocwmma::col_major>;
using FragC = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, int32_t>;
constexpr int WMMA_BN = 128, WMMA_BK = 64, WMMA_LD = WMMA_BK + 16;  // LDS rows of 64 bytes padded by 16 (16-byte aligned)
constexpr int WMMA_ELEMS = FragC::num_elements;

template<int BM, bool HILO> struct WmmaLds {
    static constexpr int A = BM * WMMA_LD, B = WMMA_BN * WMMA_LD, STAGE = A * (HILO ? 2 : 1) + B, BYTES = 2 * STAGE;
};

template<class T, int BM, bool HILO, bool BLOCKS>
__global__ void __launch_bounds__(256) gemm_s8_wmma(const int8_t* __restrict__ a, const int8_t* __restrict__ a_lo, const float* __restrict__ sa,
                                                    const int8_t* __restrict__ b, const float* __restrict__ sb, const T* __restrict__ bias,
                                                    const T* residual, const T* __restrict__ rscale, T* y, int m, int n, int k) {
    using L = WmmaLds<BM, HILO>;
    constexpr int WM = BM / 2, WN = WMMA_BN / 4, FM = WM / 16, FN = WN / 16;  // 8 warps as 2 x 4
    constexpr int A_CHUNKS = BM * WMMA_BK / 16 / 256, B_CHUNKS = WMMA_BN * WMMA_BK / 16 / 256;  // 16-byte copies per thread
    constexpr int LDO = WMMA_BN + 8;
    static_assert(BM * LDO * sizeof(T) <= L::BYTES, "the output tile must fit in the operand LDS");
    __shared__ __align__(16) int8_t lds[L::BYTES];
    __shared__ int index_tile[256];
    const int tid = threadIdx.x, warp = tid >> 5, warp_m = warp / 4, warp_n = warp % 4;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * WMMA_BN;
    const int k_tiles = (k + WMMA_BK - 1) / WMMA_BK, kblocks = (k + 127) / 128;

    // Element i of this lane's accumulator fragments is at (coords[i] / 16, coords[i] % 16) of its 16 x 16 block.
    int coords[WMMA_ELEMS];
    index_tile[tid] = tid;
    __syncthreads();
    {
        FragC f;
        rocwmma::load_matrix_sync(f, index_tile, 16, rocwmma::mem_row_major);
        #pragma unroll
        for (int i = 0; i < WMMA_ELEMS; ++i) coords[i] = f[i];
    }

    // Copy c of a tile: row c / 4, 16-byte column c % 4; out-of-range rows and K are zero.
    int4 ra[A_CHUNKS], ra_lo[HILO ? A_CHUNKS : 1], rb[B_CHUNKS];
    auto fetch = [&](int kt) {
        const int k0 = kt * WMMA_BK;
        #pragma unroll
        for (int i = 0; i < A_CHUNKS; ++i) {
            const int c = tid + i * 256, r = m0 + c / 4, col = k0 + (c % 4) * 16;
            const bool ok = r < m && col < k;
            ra[i] = ok ? *reinterpret_cast<const int4*>(a + int64_t(r) * k + col) : make_int4(0, 0, 0, 0);
            if constexpr (HILO) ra_lo[i] = ok ? *reinterpret_cast<const int4*>(a_lo + int64_t(r) * k + col) : make_int4(0, 0, 0, 0);
        }
        #pragma unroll
        for (int i = 0; i < B_CHUNKS; ++i) {
            const int c = tid + i * 256, r = n0 + c / 4, col = k0 + (c % 4) * 16;
            rb[i] = r < n && col < k ? *reinterpret_cast<const int4*>(b + int64_t(r) * k + col) : make_int4(0, 0, 0, 0);
        }
    };
    auto stash = [&](int stage) {
        int8_t* base = lds + stage * L::STAGE;
        #pragma unroll
        for (int i = 0; i < A_CHUNKS; ++i) {
            const int c = tid + i * 256, off = (c / 4) * WMMA_LD + (c % 4) * 16;
            *reinterpret_cast<int4*>(base + off) = ra[i];
            if constexpr (HILO) *reinterpret_cast<int4*>(base + L::A + off) = ra_lo[i];
        }
        #pragma unroll
        for (int i = 0; i < B_CHUNKS; ++i) {
            const int c = tid + i * 256;
            *reinterpret_cast<int4*>(base + L::A * (HILO ? 2 : 1) + (c / 4) * WMMA_LD + (c % 4) * 16) = rb[i];
        }
    };

    FragC acc[FM][FN], acc_lo[HILO ? FM : 1][HILO ? FN : 1];
    float accf[BLOCKS ? FM : 1][BLOCKS ? FN : 1][WMMA_ELEMS];
    #pragma unroll
    for (int fm = 0; fm < FM; ++fm)
        #pragma unroll
        for (int fn = 0; fn < FN; ++fn) {
            rocwmma::fill_fragment(acc[fm][fn], 0);
            if constexpr (HILO) rocwmma::fill_fragment(acc_lo[fm][fn], 0);
            if constexpr (BLOCKS)
                #pragma unroll
                for (int i = 0; i < WMMA_ELEMS; ++i) accf[fm][fn][i] = 0.f;
        }

    fetch(0);
    stash(0);
    __syncthreads();
    for (int kt = 0; kt < k_tiles; ++kt) {
        if (kt + 1 < k_tiles) fetch(kt + 1);  // in flight during the tile's MMAs
        const int8_t* tile_a = lds + (kt & 1) * L::STAGE;
        const int8_t* tile_b = tile_a + L::A * (HILO ? 2 : 1);
        #pragma unroll
        for (int kk = 0; kk < WMMA_BK; kk += 16) {
            FragB fb[FN];
            #pragma unroll
            for (int fn = 0; fn < FN; ++fn) rocwmma::load_matrix_sync(fb[fn], tile_b + (warp_n * WN + fn * 16) * WMMA_LD + kk, WMMA_LD);
            #pragma unroll
            for (int fm = 0; fm < FM; ++fm) {
                FragA fa;
                rocwmma::load_matrix_sync(fa, tile_a + (warp_m * WM + fm * 16) * WMMA_LD + kk, WMMA_LD);
                #pragma unroll
                for (int fn = 0; fn < FN; ++fn) rocwmma::mma_sync(acc[fm][fn], fa, fb[fn], acc[fm][fn]);
                if constexpr (HILO) {
                    rocwmma::load_matrix_sync(fa, tile_a + L::A + (warp_m * WM + fm * 16) * WMMA_LD + kk, WMMA_LD);
                    #pragma unroll
                    for (int fn = 0; fn < FN; ++fn) rocwmma::mma_sync(acc_lo[fm][fn], fa, fb[fn], acc_lo[fm][fn]);
                }
            }
        }
        if constexpr (BLOCKS) if (kt % 2 == 1 || kt + 1 == k_tiles) {
            // The block's exact INT32 sums (|sum| < 2^22) times the row's block scale, into the FP32 accumulators.
            #pragma unroll
            for (int fm = 0; fm < FM; ++fm) {
                float scale[WMMA_ELEMS];
                #pragma unroll
                for (int i = 0; i < WMMA_ELEMS; ++i) {
                    const int row = m0 + warp_m * WM + fm * 16 + coords[i] / 16;
                    scale[i] = row < m ? sa[int64_t(row) * kblocks + kt / 2] : 0.f;
                }
                #pragma unroll
                for (int fn = 0; fn < FN; ++fn) {
                    #pragma unroll
                    for (int i = 0; i < WMMA_ELEMS; ++i) accf[fm][fn][i] = fmaf(float(acc[fm][fn][i]), scale[i], accf[fm][fn][i]);
                    rocwmma::fill_fragment(acc[fm][fn], 0);
                }
            }
        }
        if (kt + 1 < k_tiles) stash((kt + 1) & 1);  // the other stage, which every warp finished reading before the last barrier
        __syncthreads();
    }

    // Epilogue: values (bias, residual, one rounding) staged as a row-major tile in LDS, then stored by whole rows. The residual is
    // read before the tile is written: y may be the residual.
    T* tile = reinterpret_cast<T*>(lds);
    #pragma unroll
    for (int fm = 0; fm < FM; ++fm)
        #pragma unroll
        for (int fn = 0; fn < FN; ++fn)
            #pragma unroll
            for (int i = 0; i < WMMA_ELEMS; ++i) {
                const int r = warp_m * WM + fm * 16 + coords[i] / 16, c = warp_n * WN + fn * 16 + coords[i] % 16;
                const int row = m0 + r, col = n0 + c;
                float v;
                if constexpr (BLOCKS) v = accf[fm][fn][i]; else v = float(acc[fm][fn][i]);
                if constexpr (HILO) v = fmaf(float(acc_lo[fm][fn][i]), 1.f / 254.f, v);
                float out = 0.f;
                if (row < m && col < n) {
                    out = fmaf(v * (BLOCKS ? 1.f : sa[row]), sb[col], bias ? to_float(bias[col]) : 0.f);
                    if (residual) out = fmaf(out, rscale ? to_float(rscale[col]) : 1.f, to_float(residual[int64_t(row) * n + col]));
                }
                tile[r * LDO + c] = T(out);
            }
    __syncthreads();
    for (int i = tid; i < BM * WMMA_BN; i += 256) {
        const int r = i / WMMA_BN, c = i % WMMA_BN, row = m0 + r, col = n0 + c;
        if (row < m && col < n) y[int64_t(row) * n + col] = tile[r * LDO + c];
    }
}

template<class T, int BM, bool HILO, bool BLOCKS>
int launch_gemm_wmma(const int8_t* a, const int8_t* a_lo, const float* sa, const int8_t* b, const float* sb, const void* bias, const void* residual,
                     const void* rscale, void* y, int m, int n, int k, cudaStream_t stream) {
    const dim3 grid((n + WMMA_BN - 1) / WMMA_BN, (m + BM - 1) / BM);
    gemm_s8_wmma<T, BM, HILO, BLOCKS><<<grid, 256, 0, stream>>>(a, a_lo, sa, b, sb, static_cast<const T*>(bias), static_cast<const T*>(residual),
                                                                static_cast<const T*>(rscale), static_cast<T*>(y), m, n, k);
    return int(cudaGetLastError());
}
#endif  // WGP_AMD


// ---------------------------------------------------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------------------------------------------------
#if defined(_WIN32) && !defined(__HIP_DEVICE_COMPILE__)  // HIP's device pass has no DLL exports
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

API int wq8_abi() { return 10; }

template<class T, int GROUP, bool HILO, int ACT>
int launch_quantize(const void* x, int8_t* hi, int8_t* lo, float* scale, int m, int k, int64_t ldx, const void* act_weight, float eps, cudaStream_t stream) {
    auto src = static_cast<const T*>(x);
    auto aw = static_cast<const T*>(act_weight);
    if (k <= 2048) quantize_rows_registers<T, GROUP, HILO, 1, ACT><<<m, 256, 0, stream>>>(src, hi, lo, scale, k, ldx, aw, eps);
    else if (k <= 4096) quantize_rows_registers<T, GROUP, HILO, 2, ACT><<<m, 256, 0, stream>>>(src, hi, lo, scale, k, ldx, aw, eps);
    else if (k <= 8192) quantize_rows_registers<T, GROUP, HILO, 4, ACT><<<m, 256, 0, stream>>>(src, hi, lo, scale, k, ldx, aw, eps);
    else if (k <= 16384) quantize_rows_registers<T, GROUP, HILO, 8, ACT><<<m, 256, 0, stream>>>(src, hi, lo, scale, k, ldx, aw, eps);
    else quantize_rows<T, GROUP, HILO, ACT><<<m, 256, 0, stream>>>(src, hi, lo, scale, k, ldx, aw, eps);
    return int(cudaGetLastError());
}

template<class T, bool HILO, int ACT>
int dispatch_quantize(const void* x, int8_t* hi, int8_t* lo, float* scale, int m, int k, int64_t ldx, int group, const void* aw, float eps, cudaStream_t stream) {
    switch (group) {
        case 0: return launch_quantize<T, 0, HILO, ACT>(x, hi, lo, scale, m, k, ldx, aw, eps, stream);
        case 16: return launch_quantize<T, 16, HILO, ACT>(x, hi, lo, scale, m, k, ldx, aw, eps, stream);
        case 64: return launch_quantize<T, 64, HILO, ACT>(x, hi, lo, scale, m, k, ldx, aw, eps, stream);
        case 256: return launch_quantize<T, 256, HILO, ACT>(x, hi, lo, scale, m, k, ldx, aw, eps, stream);
    }
    return -1;
}

template<class T, bool HILO>
int dispatch_quantize_act(const void* x, int8_t* hi, int8_t* lo, float* scale, int m, int k, int64_t ldx, int group, int act, const void* aw, float eps,
                          cudaStream_t stream) {
    switch (act) {
        case ACT_NONE: return dispatch_quantize<T, HILO, ACT_NONE>(x, hi, lo, scale, m, k, ldx, group, aw, eps, stream);
        case ACT_RMS: return dispatch_quantize<T, HILO, ACT_RMS>(x, hi, lo, scale, m, k, ldx, group, aw, eps, stream);
        case ACT_SWIGLU: return dispatch_quantize<T, HILO, ACT_SWIGLU>(x, hi, lo, scale, m, k, ldx, group, aw, eps, stream);
    }
    return -1;
}

// Row INT8 of op(x): dtype 0 = FP16, 1 = BF16, 2 = FP32 (no input operation). K must be a multiple of 8 and of the group; rows are
// 16-byte aligned. act: 0 none, 1 RMSNorm (act_weight: K values of x's dtype or null), 2 SwiGLU (x rows hold 2K values: gate | value).
API int wq8_quantize(const void* x, int8_t* hi, int8_t* lo, float* scale, int m, int k, int64_t ldx, int dtype, int group, int act,
                     const void* act_weight, float eps, cudaStream_t stream) {
    if (!m) return 0;
    if (dtype == 0) return lo ? dispatch_quantize_act<__half, true>(x, hi, lo, scale, m, k, ldx, group, act, act_weight, eps, stream)
                              : dispatch_quantize_act<__half, false>(x, hi, lo, scale, m, k, ldx, group, act, act_weight, eps, stream);
    if (dtype == 1) return lo ? dispatch_quantize_act<__nv_bfloat16, true>(x, hi, lo, scale, m, k, ldx, group, act, act_weight, eps, stream)
                              : dispatch_quantize_act<__nv_bfloat16, false>(x, hi, lo, scale, m, k, ldx, group, act, act_weight, eps, stream);
    if (act) return -1;
    return lo ? dispatch_quantize<float, true, ACT_NONE>(x, hi, lo, scale, m, k, ldx, group, nullptr, 0.f, stream)
              : dispatch_quantize<float, false, ACT_NONE>(x, hi, lo, scale, m, k, ldx, group, nullptr, 0.f, stream);
}

#if !WGP_AMD
// Tile configurations: 0 = 128x256 (8 warps of 64x64), 2 = 128x128 (8 warps of 64x32), 6 = 64x128 (8 warps of 32x32);
// 1 and 7 are the 128x128 and 64x128 HILO tiles (two accumulators), 10 and 11 the 128x128 and 64x128 per-block-scale tiles. CTAS: CTAs per SM (two 4-warp 128x128 CTAs, 3 or 4 stages
// and 3-stage 128x256 tiles were measured slower).
template<int V> struct Variant;
template<> struct Variant<0> { static constexpr int BM = 128, BN = 256, BK = 64, WM = 64, WN = 64, STAGES = 4, CTAS = 1; static constexpr bool HILO = false, BLOCKS = false; };
template<> struct Variant<1> { static constexpr int BM = 128, BN = 128, BK = 64, WM = 64, WN = 32, STAGES = 4, CTAS = 1; static constexpr bool HILO = true, BLOCKS = false; };
template<> struct Variant<2> { static constexpr int BM = 128, BN = 128, BK = 64, WM = 64, WN = 32, STAGES = 4, CTAS = 1; static constexpr bool HILO = false, BLOCKS = false; };
template<> struct Variant<6> { static constexpr int BM = 64, BN = 128, BK = 64, WM = 32, WN = 32, STAGES = 4, CTAS = 1; static constexpr bool HILO = false, BLOCKS = false; };
template<> struct Variant<10> { static constexpr int BM = 128, BN = 128, BK = 64, WM = 64, WN = 32, STAGES = 4, CTAS = 1; static constexpr bool HILO = false, BLOCKS = true; };
template<> struct Variant<11> { static constexpr int BM = 64, BN = 128, BK = 64, WM = 32, WN = 32, STAGES = 4, CTAS = 1; static constexpr bool HILO = false, BLOCKS = true; };
template<> struct Variant<7> { static constexpr int BM = 64, BN = 128, BK = 64, WM = 32, WN = 32, STAGES = 4, CTAS = 1; static constexpr bool HILO = true, BLOCKS = false; };
// Turing (sm_75, 64 KiB of shared memory per CTA): register-staged copies (a k-tile is read into registers a tile ahead and stored at
// the end of the next one). The 128x128 and 64x128 tiles have 3 stages, the 128x256 tile 2 (48 KiB; a second barrier per k-tile); the
// 128x128 HILO tile (72 KiB with 3 stages) is replaced by the 64x128 one.
template<int V, int STAGES_> struct Stages : Variant<V> { static constexpr int STAGES = STAGES_; };
template<> struct Variant<100> : Stages<0, 2> {};
template<> struct Variant<102> : Stages<2, 3> {};
template<> struct Variant<106> : Stages<6, 3> {};
template<> struct Variant<107> : Stages<7, 3> {};
template<> struct Variant<110> : Stages<10, 3> {};
template<> struct Variant<111> : Stages<11, 3> {};

// Compute capability below 8.0 of the current device (cached per device).
static bool sm75_device() {
    static int known[64] = {};
    int device = 0;
    cudaGetDevice(&device);
    if (device < 0 || device >= 64) return false;
    if (!known[device]) {
        int major = 8;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
        known[device] = major < 8 ? 1 : 2;
    }
    return known[device] == 1;
}

// The variant a configuration runs as on the current device: on Turing, its 3-stage counterpart.
static int device_variant(int variant) {
    if (!sm75_device()) return variant;
    switch (variant) {
        case 0: return 100;
        case 2: return 102;
        case 6: return 106;
        case 1: case 7: return 107;
        case 10: return 110;
        case 11: return 111;
    }
    return variant;
}

// split: 0 = whole tiles only, 1 = stream-K for an incomplete last wave.
template<class V> Schedule make_schedule(int m, int n, int k, int sms, int split) {
    sms *= V::CTAS;  // concurrent CTA slots
    Schedule s{};
    s.tiles_m = (m + V::BM - 1) / V::BM; s.tiles_n = (n + V::BN - 1) / V::BN; s.k_tiles = (k + V::BK - 1) / V::BK;
    const int tiles = s.tiles_m * s.tiles_n, grid = tiles < sms ? tiles : sms, rest = tiles % grid;
    s.dp_tiles = tiles; s.sk_tiles = 0; s.sk_ctas = 0; s.sk_units = 0;
    if (split && (rest || tiles < sms)) {
        s.sk_tiles = tiles < sms ? tiles : rest;
        s.dp_tiles = tiles - s.sk_tiles;
        s.sk_units = (long long)s.sk_tiles * s.k_tiles;
        // At least four k-tiles and at most four pieces per tile: the last piece reads the others' partial sums.
        long long ctas = s.sk_units / 4;
        if (ctas > 4LL * s.sk_tiles) ctas = 4LL * s.sk_tiles;
        s.sk_ctas = int(ctas < sms ? (ctas < 1 ? 1 : ctas) : sms);
    }
    s.grid = grid > s.sk_ctas ? grid : s.sk_ctas;
    return s;
}

// Partial-sum slots (uninitialized) and zeroed per-tile counters, which the kernel leaves zeroed.
template<int VI> void workspace_sizes(int m, int n, int k, int sms, int split, int64_t* sizes) {
    using V = Variant<VI>;
    const Schedule s = make_schedule<V>(m, n, k, sms, split);
    sizes[0] = s.sk_tiles ? int64_t(s.sk_ctas + s.sk_tiles) * V::BM * V::BN * 4 * (V::HILO ? 2 : 1) : 0;
    sizes[1] = 4 * int64_t(s.sk_tiles);
}

API void wq8_gemm_workspace(int m, int n, int k, int variant, int sms, int split, int64_t* sizes) {
    switch (device_variant(variant)) {
        case 100: return workspace_sizes<100>(m, n, k, sms, split, sizes);
        case 102: return workspace_sizes<102>(m, n, k, sms, split, sizes);
        case 106: return workspace_sizes<106>(m, n, k, sms, split, sizes);
        case 107: return workspace_sizes<107>(m, n, k, sms, split, sizes);
        case 110: return workspace_sizes<110>(m, n, k, sms, split, sizes);
        case 111: return workspace_sizes<111>(m, n, k, sms, split, sizes);
        case 1: return workspace_sizes<1>(m, n, k, sms, split, sizes);
        case 2: return workspace_sizes<2>(m, n, k, sms, split, sizes);
        case 6: return workspace_sizes<6>(m, n, k, sms, split, sizes);
        case 7: return workspace_sizes<7>(m, n, k, sms, split, sizes);
        case 10: return workspace_sizes<10>(m, n, k, sms, split, sizes);
        case 11: return workspace_sizes<11>(m, n, k, sms, split, sizes);
    }
    workspace_sizes<0>(m, n, k, sms, split, sizes);
}

template<class T, int VI>
int launch_gemm(const int8_t* a, const int8_t* a_lo, const float* sa, const int8_t* b, const float* sb, const void* bias, const void* residual,
                const void* rscale, void* y, int m, int n, int k, int sms, int split, int* workspace, int* counters, cudaStream_t stream) {
    using V = Variant<VI>;
    using C = GemmConfig<V::BM, V::BN, V::BK, V::WM, V::WN, V::STAGES, V::HILO>;
    auto kernel = gemm_s8<T, V::BM, V::BN, V::BK, V::WM, V::WN, V::STAGES, V::HILO, V::CTAS, V::BLOCKS>;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
        configured = true;
    }
    const Schedule s = make_schedule<V>(m, n, k, sms, split);
    kernel<<<s.grid, C::THREADS, C::SMEM, stream>>>(a, a_lo, sa, b, sb, static_cast<const T*>(bias), static_cast<const T*>(residual),
                                                    static_cast<const T*>(rscale), static_cast<T*>(y), m, n, k, s, workspace, counters);
    return int(cudaGetLastError());
}

// Tile choice: estimated time = padded work / (relative mainloop efficiency * busy SMs). Stream-K keeps every SM busy
// once tiles x pieces (at most four pieces of four k-tiles) cover them. Efficiencies were measured on an RTX 5090.
template<int VI> double estimate(int m, int n, int k, int sms, double efficiency) {
    using V = Variant<VI>;
    sms *= V::CTAS;
    const long long tm = (m + V::BM - 1) / V::BM, tn = (n + V::BN - 1) / V::BN, kt = (k + V::BK - 1) / V::BK, tiles = tm * tn;
    long long pieces = kt / 4 < 4 ? kt / 4 : 4;
    if (pieces < 1) pieces = 1;
    const double busy = tiles >= sms ? sms : double(tiles * pieces < sms ? tiles * pieces : sms);
    return double(tm * V::BM) * double(tn * V::BN) * double(k) / (efficiency * busy);
}

API int wq8_gemm_select(int m, int n, int k, int hilo, int sms) {
    if (hilo) return estimate<1>(m, n, k, sms, 0.95) <= estimate<7>(m, n, k, sms, 0.8) ? 1 : 7;
    const double e0 = estimate<0>(m, n, k, sms, 1.0), e2 = estimate<2>(m, n, k, sms, 0.95), e6 = estimate<6>(m, n, k, sms, 0.8);
    return e0 <= e2 && e0 <= e6 ? 0 : e2 <= e6 ? 2 : 6;
}

// Y (M, N) = (A (M, K) * sa) @ (B (N, K) * sb).T + bias, or with a residual R (M, N): Y = R + rscale * (... + bias) (rscale: N values or
// null for 1), computed in FP32 and rounded once; Y may be R. K must be a multiple of 16; A, B, R, Y contiguous. Variants 10 and 11 take
// per-block scales sa (M, ceil(K / 128)) from wq8_quantize_blocks.
// workspace / counters: wq8_gemm_workspace() sizes; counters zeroed, and left zeroed.
template<class T> int dispatch_gemm(const int8_t* a, const int8_t* a_lo, const float* sa, const int8_t* b, const float* sb, const void* bias,
                                    const void* residual, const void* rscale, void* y, int m, int n, int k, int variant, int sms, int split,
                                    int* workspace, int* counters, cudaStream_t stream) {
    switch (device_variant(variant)) {
        case 100: return launch_gemm<T, 100>(a, a_lo, sa, b, sb, bias, residual, rscale, y, m, n, k, sms, split, workspace, counters, stream);
        case 102: return launch_gemm<T, 102>(a, a_lo, sa, b, sb, bias, residual, rscale, y, m, n, k, sms, split, workspace, counters, stream);
        case 106: return launch_gemm<T, 106>(a, a_lo, sa, b, sb, bias, residual, rscale, y, m, n, k, sms, split, workspace, counters, stream);
        case 107: return launch_gemm<T, 107>(a, a_lo, sa, b, sb, bias, residual, rscale, y, m, n, k, sms, split, workspace, counters, stream);
        case 110: return launch_gemm<T, 110>(a, a_lo, sa, b, sb, bias, residual, rscale, y, m, n, k, sms, split, workspace, counters, stream);
        case 111: return launch_gemm<T, 111>(a, a_lo, sa, b, sb, bias, residual, rscale, y, m, n, k, sms, split, workspace, counters, stream);
        case 1: return launch_gemm<T, 1>(a, a_lo, sa, b, sb, bias, residual, rscale, y, m, n, k, sms, split, workspace, counters, stream);
        case 2: return launch_gemm<T, 2>(a, a_lo, sa, b, sb, bias, residual, rscale, y, m, n, k, sms, split, workspace, counters, stream);
        case 6: return launch_gemm<T, 6>(a, a_lo, sa, b, sb, bias, residual, rscale, y, m, n, k, sms, split, workspace, counters, stream);
        case 7: return launch_gemm<T, 7>(a, a_lo, sa, b, sb, bias, residual, rscale, y, m, n, k, sms, split, workspace, counters, stream);
        case 10: return launch_gemm<T, 10>(a, a_lo, sa, b, sb, bias, residual, rscale, y, m, n, k, sms, split, workspace, counters, stream);
        case 11: return launch_gemm<T, 11>(a, a_lo, sa, b, sb, bias, residual, rscale, y, m, n, k, sms, split, workspace, counters, stream);
        case 0: break;
        default: return -3;
    }
    return launch_gemm<T, 0>(a, a_lo, sa, b, sb, bias, residual, rscale, y, m, n, k, sms, split, workspace, counters, stream);
}
#else
// AMD: whole tiles (split and the workspace are unused); 0 / 2 / 1 = 128x128 tiles, 6 / 7 = 64x128 (row, HILO); per-block scales (10, 11)
// use the 64x128 tile, whose INT32 and FP32 accumulators fit the 256 registers (the 128-row one spills).
API void wq8_gemm_workspace(int m, int n, int k, int variant, int sms, int split, int64_t* sizes) { sizes[0] = sizes[1] = 0; }

API int wq8_gemm_select(int m, int n, int k, int hilo, int sms) {
    const bool wide = (int64_t((m + 127) / 128) * ((n + 127) / 128)) >= sms;
    return hilo ? (wide ? 1 : 7) : (wide ? 2 : 6);
}

template<class T> int dispatch_gemm(const int8_t* a, const int8_t* a_lo, const float* sa, const int8_t* b, const float* sb, const void* bias,
                                    const void* residual, const void* rscale, void* y, int m, int n, int k, int variant, int sms, int split,
                                    int* workspace, int* counters, cudaStream_t stream) {
    switch (variant) {
        case 0: case 2: return launch_gemm_wmma<T, 128, false, false>(a, a_lo, sa, b, sb, bias, residual, rscale, y, m, n, k, stream);
        case 6: return launch_gemm_wmma<T, 64, false, false>(a, a_lo, sa, b, sb, bias, residual, rscale, y, m, n, k, stream);
        case 1: return launch_gemm_wmma<T, 128, true, false>(a, a_lo, sa, b, sb, bias, residual, rscale, y, m, n, k, stream);
        case 7: return launch_gemm_wmma<T, 64, true, false>(a, a_lo, sa, b, sb, bias, residual, rscale, y, m, n, k, stream);
        case 10: case 11: return launch_gemm_wmma<T, 64, false, true>(a, a_lo, sa, b, sb, bias, residual, rscale, y, m, n, k, stream);
    }
    return -3;
}
#endif

API int wq8_gemm(const int8_t* a, const int8_t* a_lo, const float* sa, const int8_t* b, const float* sb, const void* bias, const void* residual,
                 const void* rscale, void* y, int m, int n, int k, int dtype, int variant, int sms, int split, int* workspace, int* counters,
                 cudaStream_t stream) {
    if (!m || !n) return 0;
    if ((variant == 1 || variant == 7) != (a_lo != nullptr)) return -2;
    if (dtype == 0) return dispatch_gemm<__half>(a, a_lo, sa, b, sb, bias, residual, rscale, y, m, n, k, variant, sms, split, workspace, counters, stream);
    if (dtype == 1) return dispatch_gemm<__nv_bfloat16>(a, a_lo, sa, b, sb, bias, residual, rscale, y, m, n, k, variant, sms, split, workspace, counters, stream);
    return -1;
}

// ---------------------------------------------------------------------------------------------------------------------
// Few tokens: K must be a multiple of 64, at most 32 tokens. splits: K split over CTAs so that every SM streams weights.
// ---------------------------------------------------------------------------------------------------------------------
#if !WGP_AMD
// Up to 8 tokens: 32 contiguous bytes per weight row and thread (K multiple of 128); 16 bytes otherwise.
static int small_vectors(int m, int k) { return m <= 8 && k % 128 == 0 ? 2 : 1; }

static int small_splits(int n, int k, int vectors, int sms) {
    const int blocks = (n + 15) / 16, chunks = k / (64 * vectors);
    int splits = (2 * sms + blocks - 1) / blocks;
    const int most = chunks / 16 > 1 ? chunks / 16 : 1;  // keep every warp busy for a few chunks
    return splits < most ? splits : most;
}

API void wq8_small_workspace(int m, int n, int k, int sms, int64_t* sizes) {
    const int blocks = (n + 15) / 16, splits = small_splits(n, k, small_vectors(m, k), sms), mt = m <= 8 ? 1 : m <= 16 ? 2 : 4;
    sizes[0] = splits > 1 ? int64_t(blocks) * splits * mt * 8 * 16 * 4 : 0;
    sizes[1] = splits > 1 ? 4 * int64_t(blocks) : 0;
}

template<class T, int MT, int V> int launch_small(const void* x, int64_t ldx, const int8_t* w, const float* sw, const void* bias, const void* residual,
                                                  const void* rscale, void* y, int m, int n, int k, int sms, float* slots, int* counters, cudaStream_t stream) {
    const dim3 grid((n + 15) / 16, small_splits(n, k, V, sms));
    w8a16_small<T, MT, 4, V><<<grid, 256, 0, stream>>>(static_cast<const T*>(x), ldx, w, sw, static_cast<const T*>(bias), static_cast<const T*>(residual),
                                                      static_cast<const T*>(rscale), static_cast<T*>(y), m, n, k, grid.y > 1 ? slots : nullptr, counters);
    return int(cudaGetLastError());
}

template<class T> int dispatch_lines(const void* x, int64_t ldx, const int8_t* w, const float* sw, const void* bias, const void* residual,
                                     const void* rscale, void* y, int m, int n, int k, cudaStream_t stream);

template<class T> int dispatch_small(const void* x, int64_t ldx, const int8_t* w, const float* sw, const void* bias, const void* residual, const void* rscale,
                                     void* y, int m, int n, int k, int sms, float* slots, int* counters, cudaStream_t stream) {
    if (std::is_same<T, __nv_bfloat16>::value && sm75_device()) {
        // Turing has no BF16 MMA: GEMVs (unquantized activations, exact INT8 weights, FP32 accumulation) of up to 7 tokens in one pass,
        // of four tokens beyond (measured on an RTX 2080 Ti: 5 rows in one pass 0.40 ms vs 0.65 ms in two, 8 rows 0.88 vs 0.69 ms)
        const int group = m <= 7 ? m : 4;
        for (int r = 0; r < m; r += group) {
            const int rows = m - r < group ? m - r : group;
            const int code = dispatch_lines<T>(static_cast<const T*>(x) + int64_t(r) * ldx, ldx, w, sw, bias,
                                               residual ? static_cast<const T*>(residual) + int64_t(r) * n : nullptr, rscale,
                                               static_cast<T*>(y) + int64_t(r) * n, rows, n, k, stream);
            if (code) return code;
        }
        return 0;
    }
    if (m <= 8 && small_vectors(m, k) == 2) return launch_small<T, 1, 2>(x, ldx, w, sw, bias, residual, rscale, y, m, n, k, sms, slots, counters, stream);
    if (m <= 8) return launch_small<T, 1, 1>(x, ldx, w, sw, bias, residual, rscale, y, m, n, k, sms, slots, counters, stream);
    if (m <= 16) return launch_small<T, 2, 1>(x, ldx, w, sw, bias, residual, rscale, y, m, n, k, sms, slots, counters, stream);
    return launch_small<T, 4, 1>(x, ldx, w, sw, bias, residual, rscale, y, m, n, k, sms, slots, counters, stream);
}
#else
API void wq8_small_workspace(int m, int n, int k, int sms, int64_t* sizes) { sizes[0] = sizes[1] = 0; }

template<class T> int dispatch_lines(const void* x, int64_t ldx, const int8_t* w, const float* sw, const void* bias, const void* residual,
                                     const void* rscale, void* y, int m, int n, int k, cudaStream_t stream);

// Up to 32 tokens as GEMVs of four tokens: unquantized activations, INT8 weights converted exactly, FP32 accumulation (the weights of
// the later groups come from the cache).
template<class T> int dispatch_small(const void* x, int64_t ldx, const int8_t* w, const float* sw, const void* bias, const void* residual, const void* rscale,
                                     void* y, int m, int n, int k, int sms, float* slots, int* counters, cudaStream_t stream) {
    for (int r = 0; r < m; r += 4) {
        const int rows = m - r < 4 ? m - r : 4;
        const int code = dispatch_lines<T>(static_cast<const T*>(x) + int64_t(r) * ldx, ldx, w, sw, bias,
                                           residual ? static_cast<const T*>(residual) + int64_t(r) * n : nullptr, rscale,
                                           static_cast<T*>(y) + int64_t(r) * n, rows, n, k, stream);
        if (code) return code;
    }
    return 0;
}
#endif

// Y (M, N) = X (M, K) @ (W (N, K) * sw).T + bias with unquantized activations; X rows 16-byte aligned (ldx in elements). Residual as wq8_gemm.
API int wq8_small(const void* x, int64_t ldx, const int8_t* w, const float* sw, const void* bias, const void* residual, const void* rscale, void* y,
                  int m, int n, int k, int dtype, int sms, float* slots, int* counters, cudaStream_t stream) {
    if (!m || !n) return 0;
    if (k % 64 || m > 32) return -2;
    if (dtype == 0) return dispatch_small<__half>(x, ldx, w, sw, bias, residual, rscale, y, m, n, k, sms, slots, counters, stream);
    if (dtype == 1) return dispatch_small<__nv_bfloat16>(x, ldx, w, sw, bias, residual, rscale, y, m, n, k, sms, slots, counters, stream);
    return -1;
}

// ---------------------------------------------------------------------------------------------------------------------
// One to four tokens, after the GEMV of the native INT8 proof of concept (shared/kernels/native_int8): one warp per output
// channel, each lane reading 4 weight bytes per step, so that every warp load is one 128-byte line of a row; 4 CTA warps.
// The token count is a template constant: a runtime row test inside the unrolled loop stops the compiler from hoisting the
// next loads and cost up to 45% at two tokens.
// ---------------------------------------------------------------------------------------------------------------------
// Exact signed INT8 byte (selected by SEL from w ^ 0x80808080) as FP32: (2^23 + (b ^ 0x80)) - (2^23 + 128).
DEVICE float int8_value(uint32_t flipped, uint32_t sel) {
    return __uint_as_float(__byte_perm(flipped, 0x4B000000u, sel)) - 8388736.f;
}

template<class T, int ROWS>
__global__ void __launch_bounds__(128) gemv_lines(const T* __restrict__ x, int64_t ldx, const int8_t* __restrict__ w, const float* __restrict__ sw,
                                                  const T* __restrict__ bias, const T* residual, const T* __restrict__ rscale, T* y,
                                                  int m, int n, int k) {
    const int lane = threadIdx.x & 31, ch = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (ch >= n) return;
    const int8_t* row = w + int64_t(ch) * k;
    float acc[ROWS][4] = {};
    #pragma unroll 4
    for (int base = lane * 4; base < k; base += 128) {
        const uint32_t f = __ldg(reinterpret_cast<const uint32_t*>(row + base)) ^ 0x80808080u;
        const float wf[4] = {int8_value(f, 0x7440), int8_value(f, 0x7441), int8_value(f, 0x7442), int8_value(f, 0x7443)};
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const uint2 raw = __ldg(reinterpret_cast<const uint2*>(x + int64_t(r) * ldx + base));
            float xv[4];
            if constexpr (std::is_same<T, __nv_bfloat16>::value) {
                xv[0] = __uint_as_float(raw.x << 16); xv[1] = __uint_as_float(raw.x & 0xffff0000u);
                xv[2] = __uint_as_float(raw.y << 16); xv[3] = __uint_as_float(raw.y & 0xffff0000u);
            } else {
                const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&raw.x)), b = __half22float2(*reinterpret_cast<const __half2*>(&raw.y));
                xv[0] = a.x; xv[1] = a.y; xv[2] = b.x; xv[3] = b.y;
            }
            #pragma unroll
            for (int j = 0; j < 4; ++j) acc[r][j] = fmaf(wf[j], xv[j], acc[r][j]);
        }
    }
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        float a = (acc[r][0] + acc[r][1]) + (acc[r][2] + acc[r][3]);
        #pragma unroll
        for (int d = 16; d; d >>= 1) a += shfl_xor(a, d);
        if (!lane) {
            float v = a * sw[ch] + (bias ? to_float(bias[ch]) : 0.f);
            if (residual) v = fmaf(v, rscale ? to_float(rscale[ch]) : 1.f, to_float(residual[int64_t(r) * n + ch]));
            y[int64_t(r) * n + ch] = T(v);
        }
    }
}

// ConvRot rotation: 256 threads per CTA, a thread per element of a 256-element chunk holding whole groups (one of 256, four of 64
// or sixteen of 16), the radix-4 stages in shared memory up to the group size. Elements past K are zero and not stored.
template<class T, int GROUP>
__global__ void __launch_bounds__(256) rotate_groups(const T* __restrict__ x, T* __restrict__ out, int k, int64_t ldx) {
    __shared__ float v[256];
    const int i = threadIdx.x, index = blockIdx.x * 256 + i;
    v[i] = index < k ? to_float(x[int64_t(blockIdx.y) * ldx + index]) : 0.f;
    __syncthreads();
    #pragma unroll
    for (int stride = 1; stride < GROUP; stride *= 4) {
        const int first = i / (4 * stride) * (4 * stride) + i % stride;
        const float s = (v[first] + v[first + stride]) + (v[first + 2 * stride] + v[first + 3 * stride]);
        const float value = s - 2.f * v[first + (3 - (i / stride) % 4) * stride];
        __syncthreads();
        v[i] = value;
        __syncthreads();
    }
    constexpr float norm = GROUP == 256 ? 1.f / 16 : GROUP == 64 ? 1.f / 8 : 1.f / 4;
    if (index < k) out[int64_t(blockIdx.y) * k + index] = T(v[i] * norm);
}

template<class T> int dispatch_lines(const void* x, int64_t ldx, const int8_t* w, const float* sw, const void* bias, const void* residual,
                                     const void* rscale, void* y, int m, int n, int k, cudaStream_t stream) {
    auto src = static_cast<const T*>(x);
    auto b = static_cast<const T*>(bias), r = static_cast<const T*>(residual), rs = static_cast<const T*>(rscale);
    auto dst = static_cast<T*>(y);
    const int grid = (n + 3) / 4;
    if (m == 1) gemv_lines<T, 1><<<grid, 128, 0, stream>>>(src, ldx, w, sw, b, r, rs, dst, m, n, k);
    else if (m == 2) gemv_lines<T, 2><<<grid, 128, 0, stream>>>(src, ldx, w, sw, b, r, rs, dst, m, n, k);
    else if (m == 3) gemv_lines<T, 3><<<grid, 128, 0, stream>>>(src, ldx, w, sw, b, r, rs, dst, m, n, k);
    else if (m == 4) gemv_lines<T, 4><<<grid, 128, 0, stream>>>(src, ldx, w, sw, b, r, rs, dst, m, n, k);
    // 5 to 7 rows: Turing's BF16 few-token path (one read of the weights; from 8 rows two 4-row passes are faster)
    else if (m == 5) gemv_lines<T, 5><<<grid, 128, 0, stream>>>(src, ldx, w, sw, b, r, rs, dst, m, n, k);
    else if (m == 6) gemv_lines<T, 6><<<grid, 128, 0, stream>>>(src, ldx, w, sw, b, r, rs, dst, m, n, k);
    else gemv_lines<T, 7><<<grid, 128, 0, stream>>>(src, ldx, w, sw, b, r, rs, dst, m, n, k);
    return int(cudaGetLastError());
}

// Y (M, N) = X @ (W * sw).T + bias for one to four tokens; K multiple of 4, rows of X 8-byte aligned. Residual as wq8_gemm.
API int wq8_gemv_lines(const void* x, int64_t ldx, const int8_t* w, const float* sw, const void* bias, const void* residual, const void* rscale, void* y,
                       int m, int n, int k, int dtype, cudaStream_t stream) {
    if (!m || !n) return 0;
    if (m > 4 || k % 4) return -2;
    if (dtype == 0) return dispatch_lines<__half>(x, ldx, w, sw, bias, residual, rscale, y, m, n, k, stream);
    if (dtype == 1) return dispatch_lines<__nv_bfloat16>(x, ldx, w, sw, bias, residual, rscale, y, m, n, k, stream);
    return -1;
}

template<class T> int dispatch_rotate_groups(const void* x, void* out, int m, int k, int64_t ldx, int group, cudaStream_t stream) {
    auto src = static_cast<const T*>(x);
    auto dst = static_cast<T*>(out);
    const dim3 grid((k + 255) / 256, m);
    if (group == 16) rotate_groups<T, 16><<<grid, 256, 0, stream>>>(src, dst, k, ldx);
    else if (group == 64) rotate_groups<T, 64><<<grid, 256, 0, stream>>>(src, dst, k, ldx);
    else if (group == 256) rotate_groups<T, 256><<<grid, 256, 0, stream>>>(src, dst, k, ldx);
    else return -1;
    return int(cudaGetLastError());
}

API int wq8_rotate_groups(const void* x, void* out, int m, int k, int64_t ldx, int dtype, int group, cudaStream_t stream) {
    if (!m) return 0;
    if (dtype == 0) return dispatch_rotate_groups<__half>(x, out, m, k, ldx, group, stream);
    if (dtype == 1) return dispatch_rotate_groups<__nv_bfloat16>(x, out, m, k, ldx, group, stream);
    return -1;
}

template<class T, int GROUP, int ACT>
int launch_quantize_blocks(const T* x, int8_t* q, float* scale, int m, int k, int64_t ldx, const T* aw, float eps, float* inv_rms, cudaStream_t stream) {
    if (ACT == ACT_RMS && k <= 16384) {
        if (k <= 2048) quantize_blocks_rms<T, GROUP, 1><<<m, 256, 0, stream>>>(x, q, scale, k, ldx, aw, eps);
        else if (k <= 4096) quantize_blocks_rms<T, GROUP, 2><<<m, 256, 0, stream>>>(x, q, scale, k, ldx, aw, eps);
        else if (k <= 8192) quantize_blocks_rms<T, GROUP, 4><<<m, 256, 0, stream>>>(x, q, scale, k, ldx, aw, eps);
        else quantize_blocks_rms<T, GROUP, 8><<<m, 256, 0, stream>>>(x, q, scale, k, ldx, aw, eps);
        return int(cudaGetLastError());
    }
    if (ACT == ACT_RMS) {
        if (!inv_rms) return -2;
        row_inv_rms<T, ACT_NONE><<<m, 256, 0, stream>>>(x, inv_rms, k, ldx, eps);
    }
    const dim3 grid((k + 2047) / 2048, m);
    quantize_blocks<T, GROUP, ACT><<<grid, 256, 0, stream>>>(x, q, scale, k, ldx, aw, inv_rms);
    return int(cudaGetLastError());
}

template<class T, int ACT>
int dispatch_quantize_blocks(const void* x, int8_t* q, float* scale, int m, int k, int64_t ldx, int group, const void* act_weight, float eps,
                             float* inv_rms, cudaStream_t stream) {
    auto src = static_cast<const T*>(x);
    auto aw = static_cast<const T*>(act_weight);
    switch (group) {
        case 0: return launch_quantize_blocks<T, 0, ACT>(src, q, scale, m, k, ldx, aw, eps, inv_rms, stream);
        case 16: return launch_quantize_blocks<T, 16, ACT>(src, q, scale, m, k, ldx, aw, eps, inv_rms, stream);
        case 64: return launch_quantize_blocks<T, 64, ACT>(src, q, scale, m, k, ldx, aw, eps, inv_rms, stream);
        case 256: return launch_quantize_blocks<T, 256, ACT>(src, q, scale, m, k, ldx, aw, eps, inv_rms, stream);
    }
    return -1;
}

template<class T>
int dispatch_quantize_blocks_act(const void* x, int8_t* q, float* scale, int m, int k, int64_t ldx, int group, int act, const void* aw, float eps,
                                 float* inv_rms, cudaStream_t stream) {
    switch (act) {
        case ACT_NONE: return dispatch_quantize_blocks<T, ACT_NONE>(x, q, scale, m, k, ldx, group, aw, eps, inv_rms, stream);
        case ACT_RMS: return dispatch_quantize_blocks<T, ACT_RMS>(x, q, scale, m, k, ldx, group, aw, eps, inv_rms, stream);
        case ACT_SWIGLU: return dispatch_quantize_blocks<T, ACT_SWIGLU>(x, q, scale, m, k, ldx, group, aw, eps, inv_rms, stream);
    }
    return -1;
}

// Per-block INT8 of op(x) (optionally ConvRot-rotated): q (M, K), scale (M, ceil(K / 128)). M at most 65535 per call. act as wq8_quantize;
// RMSNorm with K > 16384 needs inv_rms scratch (M floats).
API int wq8_quantize_blocks(const void* x, int8_t* q, float* scale, int m, int k, int64_t ldx, int dtype, int group, int act, const void* act_weight,
                            float eps, float* inv_rms, cudaStream_t stream) {
    if (!m) return 0;
    if (dtype == 0) return dispatch_quantize_blocks_act<__half>(x, q, scale, m, k, ldx, group, act, act_weight, eps, inv_rms, stream);
    if (dtype == 1) return dispatch_quantize_blocks_act<__nv_bfloat16>(x, q, scale, m, k, ldx, group, act, act_weight, eps, inv_rms, stream);
    if (act) return -1;
    return dispatch_quantize_blocks<float, ACT_NONE>(x, q, scale, m, k, ldx, group, nullptr, 0.f, nullptr, stream);
}

template<class T, int GROUP>
int launch_transform(const void* x, void* out, int m, int k, int64_t ldx, int act, const void* aw, float eps, cudaStream_t stream) {
    auto src = static_cast<const T*>(x);
    auto w = static_cast<const T*>(aw);
    auto dst = static_cast<T*>(out);
    switch (act) {
        case ACT_NONE: transform_rows<T, GROUP, ACT_NONE><<<m, 256, 0, stream>>>(src, dst, k, ldx, w, eps); break;
        case ACT_RMS: transform_rows<T, GROUP, ACT_RMS><<<m, 256, 0, stream>>>(src, dst, k, ldx, w, eps); break;
        case ACT_SWIGLU: transform_rows<T, GROUP, ACT_SWIGLU><<<m, 256, 0, stream>>>(src, dst, k, ldx, w, eps); break;
        default: return -1;
    }
    return int(cudaGetLastError());
}

template<class T> int dispatch_transform(const void* x, void* out, int m, int k, int64_t ldx, int group, int act, const void* aw, float eps, cudaStream_t stream) {
    switch (group) {
        case 0: return launch_transform<T, 0>(x, out, m, k, ldx, act, aw, eps, stream);
        case 16: return launch_transform<T, 16>(x, out, m, k, ldx, act, aw, eps, stream);
        case 64: return launch_transform<T, 64>(x, out, m, k, ldx, act, aw, eps, stream);
        case 256: return launch_transform<T, 256>(x, out, m, k, ldx, act, aw, eps, stream);
    }
    return -1;
}

// out (M, K) = rotate(op(x)) rounded once to x's dtype, for the few-token paths (unquantized activations); K multiple of 8.
API int wq8_transform_rows(const void* x, void* out, int m, int k, int64_t ldx, int dtype, int group, int act, const void* act_weight, float eps,
                           cudaStream_t stream) {
    if (!m) return 0;
    if (dtype == 0) return dispatch_transform<__half>(x, out, m, k, ldx, group, act, act_weight, eps, stream);
    if (dtype == 1) return dispatch_transform<__nv_bfloat16>(x, out, m, k, ldx, group, act, act_weight, eps, stream);
    return -1;
}

// ---------------------------------------------------------------------------------------------------------------------
// L2 eviction for timing: reading a buffer larger than the L2 cache evicts the weights, as in a model whose layers stream their
// weights from memory. The buffer comes from cudaMalloc, outside PyTorch's allocator, so that tuning during a CUDA graph's eager
// warmup (MMGP runs the first call of a block eagerly before capturing it) cannot leave it in the graph's memory pool.
// ---------------------------------------------------------------------------------------------------------------------
__global__ void l2_flush_read(const int4* __restrict__ p, int64_t n, int* sink) {
    int4 acc = make_int4(0, 0, 0, 0);
    for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        const int4 v = __ldcg(p + i);
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x5bd1e995 && sink) *sink = 1;  // keeps the reads; practically never stores
}

static void* g_flush = nullptr;
static int64_t g_flush_bytes = 0;

// Reads `bytes` (allocated on first use, on `device`) on the stream; returns nonzero when the memory cannot be allocated.
API int wq8_l2_flush(int device, int64_t bytes, cudaStream_t stream) {
    if (g_flush_bytes < bytes) {
        (void)cudaSetDevice(device);
        if (g_flush) (void)cudaFree(g_flush);
        g_flush = nullptr; g_flush_bytes = 0;
        if (cudaMalloc(&g_flush, bytes + 16) != cudaSuccess) {
            (void)cudaGetLastError();
            g_flush = nullptr;
            return 1;
        }
        g_flush_bytes = bytes;
    }
    l2_flush_read<<<1024, 256, 0, stream>>>(static_cast<const int4*>(g_flush), bytes / 16, static_cast<int*>(g_flush) + bytes / 4);
    return int(cudaGetLastError());
}

API void wq8_l2_flush_release() {
    if (g_flush) (void)cudaFree(g_flush);
    g_flush = nullptr; g_flush_bytes = 0;
}
