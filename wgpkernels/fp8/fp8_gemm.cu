// Copyright (c) 2026 DeepBeepMeep - Apache License 2.0 (see LICENSE.txt)
// FP8 (E4M3) block-scaled GEMMs for FP8 checkpoints, with the semantics of DeepGEMM's FP8 GEMM and CUTLASS's GemmBlockwise:
// Y = sum over the 128-element blocks kb of K of sa[m, kb] * sb[n / 128, kb] * (Aq[m, kb] . Bq[n, kb]) + bias, where each block's dot
// product starts from zero in FP32 and is promoted (scaled and added) into FP32 accumulators. A: (M, K) E4M3 activations with FP32
// scales per row and block (amax / 448); B: the checkpoint's (N, K) E4M3 weights with FP32 scales per 128 x 128 block or per output
// channel and block (folded with the activation scales), or per output channel or tensor (applied in the epilogue). These are approximate (W8A8) kernels: the activation quantization dominates the error.
// Tensor cores: E4M3 MMAs on SM89+ (RTX 40/50); on SM80-88 and SM75 the E4M3 bytes are decoded exactly to FP16 in registers (FP16
// MMAs with FP32 accumulation: the same products, another summation order); AMD RDNA4: rocWMMA's gfx12 FP8 WMMA.
#if defined(__HIP__)
// AMD (HIP, RDNA4 gfx12, wave32): the shared code keeps CUDA names, mapped here; the GEMM has an AMD counterpart behind the same C API.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
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
// Activation quantization: E4M3 values and one FP32 scale per row and 128-element block of K.
// ---------------------------------------------------------------------------------------------------------------------
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

// E4M3 byte of f, rounded to nearest even as PyTorch's .to(torch.float8_e4m3fn) (c10's fp8e4m3fn_from_fp32_value) and saturated to
// +/-448 where PyTorch returns NaN (|f| > 464); NaN stays NaN.
DEVICE uint32_t e4m3_from_float(float f) {
    uint32_t bits = __float_as_uint(f);
    const uint32_t sign = (bits >> 24) & 0x80u;
    bits &= 0x7fffffffu;
    uint32_t r;
    if (bits > 0x7f800000u) r = 0x7f;
    else if (bits > 0x43e80000u) r = 0x7e;
    // below 2^-6: adding 2^14 rounds to the subnormal step 2^-9 (the bits of 2^14 are 141 << 23)
    else if (bits < (121u << 23)) r = __float_as_uint(__fadd_rn(__uint_as_float(bits), 16384.f)) - (141u << 23);
    else r = (bits - (120u << 23) + 0x7ffffu + ((bits >> 20) & 1)) >> 20;
    return r | sign;
}

// The 8 values of v times factor (or divided by it) as E4M3, stored at p (8-byte aligned).
template<bool DIVIDE> DEVICE void store8_e4m3(uint8_t* p, const float* v, float factor) {
    uint32_t w[2];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        uint32_t r = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) r |= e4m3_from_float(DIVIDE ? v[4 * h + j] / factor : v[4 * h + j] * factor) << (8 * j);
        w[h] = r;
    }
    *reinterpret_cast<uint2*>(p) = make_uint2(w[0], w[1]);
}

// One scale per row and block, as DeepGEMM's per_token_cast_to_fp8 in PyTorch: amax clamped to 1e-4, scale = amax / 448, q = e4m3(x * (1 / scale)).
// A warp per 256-element chunk (a half warp per block); the chunks of a row are in consecutive warps.
template<class T>
__global__ void __launch_bounds__(256) quantize_rows_f8(const T* __restrict__ x, uint8_t* __restrict__ q, float* __restrict__ scale, int m, int k,
                                                        int64_t ldx, int chunks) {
    const int lane = threadIdx.x & 31;
    const int64_t item = int64_t(blockIdx.x) * 8 + (threadIdx.x >> 5);
    const int row = int(item / chunks), base = int(item % chunks) * 256;
    if (row >= m) return;  // warp-uniform
    const int col = base + lane * 8;
    const bool valid = col < k;
    float v[8];
    load8(x + int64_t(row) * ldx + col, v, valid);
    float amax = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(v[j]));
    #pragma unroll
    for (int d = 8; d; d >>= 1) amax = fmaxf(amax, shfl_xor(amax, d));  // within each half warp
    const float s = fmaxf(amax, 1e-4f) * (1.f / 448.f);  // PyTorch divides by a scalar as a multiplication by its reciprocal
    const int kblocks = (k + 127) / 128, block = base / 128 + (lane >> 4);
    if ((lane & 15) == 0 && block < kblocks) scale[int64_t(row) * kblocks + block] = s;
    if (valid) store8_e4m3<false>(q + int64_t(row) * k + col, v, 1.f / s);
}

// One scale per 128 x 128 tile, as the activation quantization of upstream's sm89 CUTLASS GemmBlockwise path: amax over the tile's rows
// and block clamped to 1e-4, scale = amax / 448, q = e4m3(x / scale); the scale is stored for every row of the tile (the GEMM takes row
// scales). A CTA per (block, 128-row tile); step i of warp w holds rows 16 i + 2 w (lanes 0-15) and 16 i + 2 w + 1 (lanes 16-31).
template<class T>
__global__ void __launch_bounds__(256) quantize_tiles_f8(const T* __restrict__ x, uint8_t* __restrict__ q, float* __restrict__ scale, int m, int k,
                                                         int64_t ldx) {
    __shared__ float warp_max[8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int kb = blockIdx.x, row0 = blockIdx.y * 128 + 2 * warp + (lane >> 4), col = kb * 128 + (lane & 15) * 8;
    float v[8][8], amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = row0 + 16 * i;
        load8(x + int64_t(row < m ? row : 0) * ldx + col, v[i], row < m && col < k);
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
    const float s = fmaxf(amax, 1e-4f) * (1.f / 448.f);  // PyTorch divides by a scalar as a multiplication by its reciprocal
    const int kblocks = (k + 127) / 128;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = row0 + 16 * i;
        if (row >= m) break;
        if ((lane & 15) == 0) scale[int64_t(row) * kblocks + kb] = s;
        if (col < k) store8_e4m3<true>(q + int64_t(row) * k + col, v[i], s);
    }
}

// Weight scale granularity: per 128 x 128 block or per output channel and 128-element block of K (both folded with the activation scales
// every block), per output channel or per tensor (epilogue).
enum { WS_BLOCK = 0, WS_CHANNEL = 1, WS_TENSOR = 2, WS_GROUP = 3 };

#if !WGP_AMD
// ---------------------------------------------------------------------------------------------------------------------
// PTX helpers
// ---------------------------------------------------------------------------------------------------------------------
DEVICE uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

// Turing (sm_75) has no asynchronous copies and no m16n8k16 MMA: copies are synchronous 16-byte loads and stores (the pipeline's waits
// are no-ops, its barriers keep the order), and each m16n8k16 is two m16n8k8 on the same fragment registers.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#define WGP_SM75 1
#else
#define WGP_SM75 0
#endif
// E4M3 tensor cores (mma.sync with .e4m3 operands): SM89 and later; the FP16 emulation is compiled for the earlier architectures only.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890
#define WGP_FP8_MMA 1
#else
#define WGP_FP8_MMA 0
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
// 4-byte copy (scales): .ca, the cp.async form that takes 4 bytes.
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

// E4M3 m16n8k32 with FP32 accumulation; its fragments have the layout of the INT8 m16n8k32 ones.
DEVICE void mma_e4m3(float* c, const uint32_t* a, const uint32_t* b) {
#if WGP_FP8_MMA
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    __trap();
#endif
}

// Exact FP16 pairs of the E4M3 bytes 0 and 2 (even) or 1 and 3 (odd) of w, times 2^-8: the sign, then the 4 exponent and 3 mantissa bits
// as the low FP16 exponent and high mantissa bits. E4M3 subnormals become FP16 subnormals; the GEMM folds 2^16 into the block scales.
DEVICE uint32_t e4m3_even(uint32_t w) { return ((w << 7) & 0x3f803f80u) | ((w << 8) & 0x80008000u); }
DEVICE uint32_t e4m3_odd(uint32_t w) { return ((w >> 1) & 0x3f803f80u) | (w & 0x80008000u); }

// m16n8k16 FP16 MMA with FP32 accumulation (Turing: two m16n8k8 on the k 0-7 / 8-15 fragments). Not volatile: the compiler interleaves
// the MMAs of independent accumulators.
DEVICE void mma_f16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
#if WGP_SM75
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(b0));
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a2), "r"(a3), "r"(b1));
#else
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

// ---------------------------------------------------------------------------------------------------------------------
// E4M3 x E4M3 tensor-core GEMM with per-block scales, multistage cp.async pipeline, XOR-swizzled shared memory (the INT8 GEMM's
// block mode with floating-point block sums). A: (M, K) row-major E4M3 activations with scales sa (M, ceil(K / 128)). B: (N, K)
// row-major E4M3 weights with scales sb (ceil(N / 128), ceil(K / 128)), (N, ceil(K / 128)), (N) or (1).
// Each block of 128 K is accumulated from zero by the MMAs, then promoted: accf += block sums * sa[row] * sb[block] (FP32 FMAs), with
// sb[block] the scale of the tile's 128 x 128 weight block, or of each column (WS_GROUP).
//
// Persistent CTAs process whole tiles in waves. The tiles of an incomplete last wave are split along K over the CTAs (stream-K): each
// piece stores its FP32 partial sums in its own workspace slot and the last piece to arrive adds them in CTA order (its own included),
// so that the result does not depend on the arrival order; no CTA ever waits and no floating-point atomics are used.
//
// MATH_E4M3: E4M3 MMAs (SM89+). MATH_F16: each m16n8k32 E4M3 MMA is two FP16 m16n8k16 MMAs on the exactly decoded bytes. The fragment
// bytes of lane t hold k 4t..4t+3 and 4t+16..4t+19 for both A and B: the first MMA takes the k 0-15 registers, the even bytes as its
// logical k (2t, 2t+1) and the odd ones as (2t+8, 2t+9), the second MMA the k 16-31 registers. A and B use the same assignment, so every
// product of the block is summed once.
// ---------------------------------------------------------------------------------------------------------------------
enum { MATH_E4M3 = 0, MATH_F16 = 1 };

template<int BM, int BN, int BK, int WM, int WN, int STAGES>
struct GemmConfig {
    static constexpr int WARPS_M = BM / WM, WARPS_N = BN / WN, THREADS = 32 * WARPS_M * WARPS_N;
    static constexpr int MI = WM / 16, NI = WN / 8, CHUNKS = BK / 16, KSTEPS = BK / 32;
    static constexpr int A_BYTES = BM * BK, B_BYTES = BN * BK, STAGE_BYTES = A_BYTES + B_BYTES;
    static constexpr int SMEM = STAGES * STAGE_BYTES;
    static constexpr int ROW_STEP = THREADS / CHUNKS, A_PER = BM / ROW_STEP, B_PER = BN / ROW_STEP;
    static constexpr int KPB = 128 / BK;  // k-tiles per scale block
    // Ring of block scales: the slot of block b is refilled for block b + RING by the stage issued STAGES - 1 k-tiles ahead, which
    // must come after the last k-tile of block b (whose fold reads the slot).
    static constexpr int RING = 2 + (STAGES - 2) / KPB;
    static constexpr int WBLOCKS = (BN + 127) / 128;  // weight scale blocks per tile (WS_BLOCK); WS_GROUP: BN column scales
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

template<class T, int BM, int BN, int BK, int WM, int WN, int STAGES, int MATH>
DEVICE void gemm_f8_body(const uint8_t* __restrict__ a, const float* __restrict__ sa, const uint8_t* __restrict__ b, const float* __restrict__ sb,
                         int wmode, const T* __restrict__ bias, const T* residual, const T* __restrict__ rscale, T* y, int m, int n, int k,
                         const Schedule& sched, float* __restrict__ workspace, int* __restrict__ counters) {
    using C = GemmConfig<BM, BN, BK, WM, WN, STAGES>;
    constexpr int MI = C::MI, NI = C::NI, KPB = C::KPB;
    // FP16 emulation: the decoded bytes are the E4M3 values times 2^-8, their products times 2^-16.
    constexpr float UNIT = MATH == MATH_F16 ? 65536.f : 1.f;
    extern __shared__ __align__(128) uint8_t smem[];
    // The block's activation scales of the tile's BM rows, then its weight scales of the tile's columns, come with the stage of the block's
    // first k-tile (cp.async: no registers held while in flight). With 128-element k-tiles (96 KiB of stages) this ring fills the 99 KiB
    // a CTA may have on SM86/SM89/SM120: the kernel has no other shared variable.
    __shared__ __align__(16) float block_scales[C::RING][BM + BN];  // float2 column scale reads
    static_assert(BM + BN <= C::THREADS, "one thread per scale of a block");
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int warp_m = warp / C::WARPS_N, warp_n = warp % C::WARPS_N;
    const uint32_t base = smem_u32(smem);
    const int chunk = tid % C::CHUNKS, row0 = tid / C::CHUNKS;
    const int g = lane >> 2, t4 = lane & 3;
    const int kblocks = (k + 127) / 128, nblocks = (n + 127) / 128;

    float acc[MI][NI][4];   // the current block's sums
    float accf[MI][NI][4];  // promoted sums
    int work_m0 = 0, work_n0 = 0, work_kb = 0;

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
    const uint8_t* a_src[C::A_PER]; const uint8_t* b_src[C::B_PER];
    bool a_ok[C::A_PER], b_ok[C::B_PER];
    auto setup = [&](const Work& w) {
        work_m0 = w.m0;
        work_n0 = w.n0;
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
    // The scales of k-tile kt's block, with the tile that starts the block (or the work).
    auto load_scales = [&](int kt) {
        if (kt % KPB == 0 || kt == work_kb) {
            const int kb = kt / KPB;
            float* ring = block_scales[kb % C::RING];
            if (tid < BM) {
                const int row = work_m0 + tid;
                cp_async4(smem_u32(ring + tid), row < m ? sa + int64_t(row) * kblocks + kb : sa, row < m);
            } else if (wmode == WS_BLOCK && tid < BM + C::WBLOCKS) {
                const int nb = work_n0 / 128 + (tid - BM);
                cp_async4(smem_u32(ring + tid), nb < nblocks ? sb + int64_t(nb) * kblocks + kb : sb, nb < nblocks);
            } else if (wmode == WS_GROUP && tid < BM + BN) {
                const int col = work_n0 + tid - BM;
                cp_async4(smem_u32(ring + tid), col < n ? sb + int64_t(col) * kblocks + kb : sb, col < n);
            }
        }
    };
    auto load_stage = [&](int slot, int kt) {
        const int k0 = kt * BK;
        const bool k_ok = k0 + chunk * 16 < k;
        const uint32_t sa_base = base + slot * C::STAGE_BYTES, sb_base = sa_base + C::A_BYTES;
        #pragma unroll
        for (int i = 0; i < C::A_PER; ++i) {
            const bool ok = a_ok[i] && k_ok;
            cp_async16(sa_base + swizzle<C::CHUNKS>(row0 + i * C::ROW_STEP, chunk), ok ? a_src[i] + k0 : a, ok);
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
    uint4 staged_a[C::A_PER], staged_b[C::B_PER];
    auto fetch_stage = [&](int kt) {
        const int k0 = kt * BK;
        const bool k_ok = k0 + chunk * 16 < k;
        #pragma unroll
        for (int i = 0; i < C::A_PER; ++i)
            staged_a[i] = a_ok[i] && k_ok ? __ldcg(reinterpret_cast<const uint4*>(a_src[i] + k0)) : make_uint4(0, 0, 0, 0);
        #pragma unroll
        for (int i = 0; i < C::B_PER; ++i)
            staged_b[i] = b_ok[i] && k_ok ? __ldcg(reinterpret_cast<const uint4*>(b_src[i] + k0)) : make_uint4(0, 0, 0, 0);
    };
    auto store_stage = [&](int slot) {
        const uint32_t sa_base = base + slot * C::STAGE_BYTES, sb_base = sa_base + C::A_BYTES;
        #pragma unroll
        for (int i = 0; i < C::A_PER; ++i)
            asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(sa_base + swizzle<C::CHUNKS>(row0 + i * C::ROW_STEP, chunk)),
                         "r"(staged_a[i].x), "r"(staged_a[i].y), "r"(staged_a[i].z), "r"(staged_a[i].w));
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

    uint32_t af[2][MI][4], bf[2][NI][2];
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
        const uint32_t sa_base = base + slot * C::STAGE_BYTES + a_frag[ks], sb_base = base + slot * C::STAGE_BYTES + C::A_BYTES + b_frag[ks];
        #pragma unroll
        for (int mi = 0; mi < MI; ++mi) ldmatrix_x4(af[buf][mi], sa_base + mi * ROWS16);
        #pragma unroll
        for (int ni = 0; ni < NI; ni += 2) {
            uint32_t t[4];
            ldmatrix_x4(t, sb_base + (ni / 2) * ROWS16);
            bf[buf][ni][0] = t[0]; bf[buf][ni][1] = t[1]; bf[buf][ni + 1][0] = t[2]; bf[buf][ni + 1][1] = t[3];
        }
    };
    // The MMAs of one k32 step on fragment buffer buf.
    auto mma_step = [&](int buf) {
        if constexpr (MATH == MATH_E4M3) {
            #pragma unroll
            for (int mi = 0; mi < MI; ++mi)
                #pragma unroll
                for (int ni = 0; ni < NI; ++ni) mma_e4m3(acc[mi][ni], af[buf][mi], bf[buf][ni]);
        } else {
            uint32_t bh[NI][4];
            #pragma unroll
            for (int ni = 0; ni < NI; ++ni) {
                bh[ni][0] = e4m3_even(bf[buf][ni][0]); bh[ni][1] = e4m3_odd(bf[buf][ni][0]);
                bh[ni][2] = e4m3_even(bf[buf][ni][1]); bh[ni][3] = e4m3_odd(bf[buf][ni][1]);
            }
            #pragma unroll
            for (int mi = 0; mi < MI; ++mi) {
                const uint32_t* q = af[buf][mi];  // rows g / g + 8 of k 0-15, then of k 16-31
                const uint32_t lo[4] = {e4m3_even(q[0]), e4m3_even(q[1]), e4m3_odd(q[0]), e4m3_odd(q[1])};
                const uint32_t hi[4] = {e4m3_even(q[2]), e4m3_even(q[3]), e4m3_odd(q[2]), e4m3_odd(q[3])};
                #pragma unroll
                for (int ni = 0; ni < NI; ++ni) {
                    mma_f16(acc[mi][ni], lo[0], lo[1], lo[2], lo[3], bh[ni][0], bh[ni][1]);
                    mma_f16(acc[mi][ni], hi[0], hi[1], hi[2], hi[3], bh[ni][2], bh[ni][3]);
                }
            }
        }
    };

    // Accumulate the k-tiles of w, whose prologue was issued.
    auto mainloop = [&](const Work& w) {
        #pragma unroll
        for (int mi = 0; mi < MI; ++mi)
            #pragma unroll
            for (int ni = 0; ni < NI; ++ni)
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[mi][ni][j] = accf[mi][ni][j] = 0.f;
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        load_frags(0, 0, 0);
        int slot = 0;
        for (int kt = w.kb; kt < w.ke;) {
            // The k-tiles of one scale block, then its promotion, once and unpredicated.
            const int block_end = min(w.ke, (kt / KPB + 1) * KPB);
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
                    mma_step(buf);
                }
                slot = (slot + 1) % STAGES;
            }
            // Promotion: accf += block sums * sa[row] * sb[block] (* 2^16 for the FP16 emulation); the block sums restart at zero.
            const float* ring = block_scales[((kt - 1) / KPB) % C::RING];
            const float* scales = ring + warp_m * WM + g;
            if (wmode == WS_GROUP) {
                float2 col_scales[NI];  // sb[col, kb] of this thread's two columns per fragment
                #pragma unroll
                for (int ni = 0; ni < NI; ++ni) col_scales[ni] = *reinterpret_cast<const float2*>(ring + BM + warp_n * WN + ni * 8 + t4 * 2);
                #pragma unroll
                for (int mi = 0; mi < MI; ++mi) {
                    const float s0 = scales[mi * 16] * UNIT, s1 = scales[mi * 16 + 8] * UNIT;
                    #pragma unroll
                    for (int ni = 0; ni < NI; ++ni) {
                        accf[mi][ni][0] = fmaf(acc[mi][ni][0], s0 * col_scales[ni].x, accf[mi][ni][0]);
                        accf[mi][ni][1] = fmaf(acc[mi][ni][1], s0 * col_scales[ni].y, accf[mi][ni][1]);
                        accf[mi][ni][2] = fmaf(acc[mi][ni][2], s1 * col_scales[ni].x, accf[mi][ni][2]);
                        accf[mi][ni][3] = fmaf(acc[mi][ni][3], s1 * col_scales[ni].y, accf[mi][ni][3]);
                        acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;
                    }
                }
            } else {
                const float wscale = wmode == WS_BLOCK ? ring[BM + (warp_n * WN) / 128] * UNIT : UNIT;
                #pragma unroll
                for (int mi = 0; mi < MI; ++mi) {
                    const float s0 = scales[mi * 16] * wscale, s1 = scales[mi * 16 + 8] * wscale;
                    #pragma unroll
                    for (int ni = 0; ni < NI; ++ni) {
                        accf[mi][ni][0] = fmaf(acc[mi][ni][0], s0, accf[mi][ni][0]);
                        accf[mi][ni][1] = fmaf(acc[mi][ni][1], s0, accf[mi][ni][1]);
                        accf[mi][ni][2] = fmaf(acc[mi][ni][2], s1, accf[mi][ni][2]);
                        accf[mi][ni][3] = fmaf(acc[mi][ni][3], s1, accf[mi][ni][3]);
                        acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;
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
            if (wmode == WS_CHANNEL) col_scale[ni] = make_float2(col < n ? sb[col] : 0.f, col + 1 < n ? sb[col + 1] : 0.f);
            else col_scale[ni] = wmode == WS_TENSOR ? make_float2(sb[0], sb[0]) : make_float2(1.f, 1.f);
            col_bias[ni] = make_float2(bias && col < n ? to_float(bias[col]) : 0.f, bias && col + 1 < n ? to_float(bias[col + 1]) : 0.f);
        }
        auto value = [&](int mi, int ni, int half, int e) {
            return fmaf(accf[mi][ni][half * 2 + e], e ? col_scale[ni].y : col_scale[ni].x, e ? col_bias[ni].y : col_bias[ni].x);
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
                    const float rs0 = rscale ? to_float(rscale[col]) : 1.f, rs1 = rscale && col + 1 < n ? to_float(rscale[col + 1]) : 1.f;
                    v.x = fmaf(v.x, rs0, rv.x);
                    v.y = fmaf(v.y, rs1, rv.y);
                }
            }
            return v;
        };
        constexpr bool STAGED_OUTPUT = BM * (BN + 8) * sizeof(T) <= C::SMEM;
        if constexpr (STAGED_OUTPUT) if ((n & 7) == 0) {
            // Like CUTLASS's epilogue: stage the tile in shared memory (free after the mainloop; rows padded by 16 bytes so that the 8 rows
            // of a fragment hit distinct banks), then store whole rows with coalesced 16-byte accesses.
            constexpr int LD = BN + 8, VECS = BN / 8;
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

    // Split tiles: each piece stores its FP32 partial sums in its own slot; the last to arrive adds every piece in CTA order.
    auto finish = [&](const Work& w) {
        if (w.j >= 0) {
            const int first = sk_cta_of((long long)w.j * sched.k_tiles, sched), last = sk_cta_of((long long)(w.j + 1) * sched.k_tiles - 1, sched);
            if (first != last) {
                // Slot blockIdx.x + j is unique: CTA ranges are increasing, so no other CTA/tile pair has the same sum.
                constexpr int VECS = MI * NI;
                float4* slots = reinterpret_cast<float4*>(workspace);
                float4* own = slots + int64_t(blockIdx.x + w.j) * VECS * C::THREADS + tid;
                #pragma unroll
                for (int mi = 0; mi < MI; ++mi)
                    #pragma unroll
                    for (int ni = 0; ni < NI; ++ni) __stcg(own + (mi * NI + ni) * C::THREADS, *reinterpret_cast<const float4*>(accf[mi][ni]));
                __threadfence();
                __syncthreads();  // every partial sum of this piece is stored before it is counted
                if (!__syncthreads_or(!tid && atomicAdd(counters + w.j, 1) == last - first)) return;  // only the last piece goes on
                __threadfence();
                float sum[MI][NI][4] = {};
                for (int c = first; c <= last; ++c) {
                    const float4* piece = slots + int64_t(c + w.j) * VECS * C::THREADS + tid;
                    #pragma unroll
                    for (int mi = 0; mi < MI; ++mi)
                        #pragma unroll
                        for (int ni = 0; ni < NI; ++ni) {
                            const float4 v = c == int(blockIdx.x) ? *reinterpret_cast<const float4*>(accf[mi][ni]) : __ldcg(piece + (mi * NI + ni) * C::THREADS);
                            sum[mi][ni][0] += v.x; sum[mi][ni][1] += v.y; sum[mi][ni][2] += v.z; sum[mi][ni][3] += v.w;
                        }
                }
                #pragma unroll
                for (int mi = 0; mi < MI; ++mi)
                    #pragma unroll
                    for (int ni = 0; ni < NI; ++ni)
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) accf[mi][ni][j] = sum[mi][ni][j];
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

template<class T, int BM, int BN, int BK, int WM, int WN, int STAGES, int CTAS, int MATH>
__global__ void __launch_bounds__(GemmConfig<BM, BN, BK, WM, WN, STAGES>::THREADS, CTAS)
gemm_f8(const uint8_t* __restrict__ a, const float* __restrict__ sa, const uint8_t* __restrict__ b, const float* __restrict__ sb, int wmode,
        const T* __restrict__ bias, const T* residual, const T* __restrict__ rscale, T* y, int m, int n, int k, Schedule sched,
        float* __restrict__ workspace, int* __restrict__ counters) {
    // Each architecture's code holds one math: E4M3 MMAs from SM89, the FP16 emulation below (the host launches the math of the code the
    // driver loaded for the device).
    if constexpr ((MATH == MATH_E4M3) != bool(WGP_FP8_MMA)) __trap();
    else gemm_f8_body<T, BM, BN, BK, WM, WN, STAGES, MATH>(a, sa, b, sb, wmode, bias, residual, rscale, y, m, n, k, sched, workspace, counters);
}

// Its attributes tell the virtual architecture of the code the driver loaded for the device (SASS of the device, or PTX it compiled).
__global__ void code_arch_probe() {}
#endif  // !WGP_AMD

#if WGP_AMD
// ---------------------------------------------------------------------------------------------------------------------
// AMD RDNA4 (gfx12, wave32) FP8 GEMM on rocWMMA 16x16x16 E4M3 x E4M3 -> FP32 fragments (OCP E4M3, the format of the checkpoints): the
// operands, scales and epilogue of gemm_f8 as whole BM x 128 tiles (no stream-K), operands staged through registers into double-buffered
// LDS, each 128-element block accumulated from zero and promoted with its scales. The row and column of every accumulator element are
// read from rocWMMA itself (an accumulator fragment loaded from a tile of indices), not assumed from the instruction layout.
// ---------------------------------------------------------------------------------------------------------------------
using FragA = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, rocwmma::float8_t, rocwmma::row_major>;
using FragB = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, rocwmma::float8_t, rocwmma::col_major>;
using FragC = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;
using FragI = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, int32_t>;
constexpr int WMMA_BN = 128, WMMA_BK = 64, WMMA_LD = WMMA_BK + 16;  // LDS rows of 64 bytes padded by 16 (16-byte aligned)
constexpr int WMMA_ELEMS = FragC::num_elements;
static_assert(FragI::num_elements == WMMA_ELEMS, "index and value accumulators must have the same layout");

template<int BM> struct WmmaLds {
    static constexpr int A = BM * WMMA_LD, B = WMMA_BN * WMMA_LD, STAGE = A + B, BYTES = 2 * STAGE;
};

template<class T, int BM>
__global__ void __launch_bounds__(256) gemm_f8_wmma(const uint8_t* __restrict__ a, const float* __restrict__ sa, const uint8_t* __restrict__ b,
                                                    const float* __restrict__ sb, int wmode, const T* __restrict__ bias, const T* residual,
                                                    const T* __restrict__ rscale, T* y, int m, int n, int k) {
    using L = WmmaLds<BM>;
    constexpr int WM = BM / 2, WN = WMMA_BN / 4, FM = WM / 16, FN = WN / 16;  // 8 warps as 2 x 4
    constexpr int A_CHUNKS = BM * WMMA_BK / 16 / 256, B_CHUNKS = WMMA_BN * WMMA_BK / 16 / 256;  // 16-byte copies per thread
    constexpr int LDO = WMMA_BN + 8;
    static_assert(BM * LDO * sizeof(T) <= L::BYTES, "the output tile must fit in the operand LDS");
    __shared__ __align__(16) uint8_t lds[L::BYTES];
    __shared__ int index_tile[256];
    const int tid = threadIdx.x, warp = tid >> 5, warp_m = warp / 4, warp_n = warp % 4;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * WMMA_BN;
    const int k_tiles = (k + WMMA_BK - 1) / WMMA_BK, kblocks = (k + 127) / 128;

    // Element i of this lane's accumulator fragments is at (coords[i] / 16, coords[i] % 16) of its 16 x 16 block.
    int coords[WMMA_ELEMS];
    index_tile[tid] = tid;
    __syncthreads();
    {
        FragI f;
        rocwmma::load_matrix_sync(f, index_tile, 16, rocwmma::mem_row_major);
        #pragma unroll
        for (int i = 0; i < WMMA_ELEMS; ++i) coords[i] = f[i];
    }

    // Copy c of a tile: row c / 4, 16-byte column c % 4; out-of-range rows and K are zero.
    int4 ra[A_CHUNKS], rb[B_CHUNKS];
    auto fetch = [&](int kt) {
        const int k0 = kt * WMMA_BK;
        #pragma unroll
        for (int i = 0; i < A_CHUNKS; ++i) {
            const int c = tid + i * 256, r = m0 + c / 4, col = k0 + (c % 4) * 16;
            ra[i] = r < m && col < k ? *reinterpret_cast<const int4*>(a + int64_t(r) * k + col) : make_int4(0, 0, 0, 0);
        }
        #pragma unroll
        for (int i = 0; i < B_CHUNKS; ++i) {
            const int c = tid + i * 256, r = n0 + c / 4, col = k0 + (c % 4) * 16;
            rb[i] = r < n && col < k ? *reinterpret_cast<const int4*>(b + int64_t(r) * k + col) : make_int4(0, 0, 0, 0);
        }
    };
    auto stash = [&](int stage) {
        uint8_t* base = lds + stage * L::STAGE;
        #pragma unroll
        for (int i = 0; i < A_CHUNKS; ++i) {
            const int c = tid + i * 256;
            *reinterpret_cast<int4*>(base + (c / 4) * WMMA_LD + (c % 4) * 16) = ra[i];
        }
        #pragma unroll
        for (int i = 0; i < B_CHUNKS; ++i) {
            const int c = tid + i * 256;
            *reinterpret_cast<int4*>(base + L::A + (c / 4) * WMMA_LD + (c % 4) * 16) = rb[i];
        }
    };

    FragC acc[FM][FN];
    float accf[FM][FN][WMMA_ELEMS];
    #pragma unroll
    for (int fm = 0; fm < FM; ++fm)
        #pragma unroll
        for (int fn = 0; fn < FN; ++fn) {
            rocwmma::fill_fragment(acc[fm][fn], 0.f);
            #pragma unroll
            for (int i = 0; i < WMMA_ELEMS; ++i) accf[fm][fn][i] = 0.f;
        }

    fetch(0);
    stash(0);
    __syncthreads();
    for (int kt = 0; kt < k_tiles; ++kt) {
        if (kt + 1 < k_tiles) fetch(kt + 1);  // in flight during the tile's MMAs
        const auto* tile_a = reinterpret_cast<const rocwmma::float8_t*>(lds + (kt & 1) * L::STAGE);
        const auto* tile_b = tile_a + L::A;
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
            }
        }
        if (kt % 2 == 1 || kt + 1 == k_tiles) {
            // Promotion of the block's FP32 sums with the row's block scale and the tile's weight block scale (BN = 128: one block), or
            // each column's (WS_GROUP).
            const int kb = kt / 2;
            const float wscale = wmode == WS_BLOCK ? sb[int64_t(n0 / 128) * kblocks + kb] : 1.f;
            #pragma unroll
            for (int fm = 0; fm < FM; ++fm) {
                float scale[WMMA_ELEMS];
                #pragma unroll
                for (int i = 0; i < WMMA_ELEMS; ++i) {
                    const int row = m0 + warp_m * WM + fm * 16 + coords[i] / 16;
                    scale[i] = row < m ? sa[int64_t(row) * kblocks + kb] * wscale : 0.f;
                }
                #pragma unroll
                for (int fn = 0; fn < FN; ++fn) {
                    #pragma unroll
                    for (int i = 0; i < WMMA_ELEMS; ++i) {
                        float s = scale[i];
                        if (wmode == WS_GROUP) {
                            const int col = n0 + warp_n * WN + fn * 16 + coords[i] % 16;
                            s *= col < n ? sb[int64_t(col) * kblocks + kb] : 0.f;
                        }
                        accf[fm][fn][i] = fmaf(acc[fm][fn][i], s, accf[fm][fn][i]);
                    }
                    rocwmma::fill_fragment(acc[fm][fn], 0.f);
                }
            }
        }
        if (kt + 1 < k_tiles) stash((kt + 1) & 1);  // the other stage, which every warp finished reading before the last barrier
        __syncthreads();
    }

    // Epilogue: values (weight scales, bias, residual, one rounding) staged as a row-major tile in LDS, then stored by whole rows. The
    // residual is read before the tile is written: y may be the residual.
    T* tile = reinterpret_cast<T*>(lds);
    #pragma unroll
    for (int fm = 0; fm < FM; ++fm)
        #pragma unroll
        for (int fn = 0; fn < FN; ++fn)
            #pragma unroll
            for (int i = 0; i < WMMA_ELEMS; ++i) {
                const int r = warp_m * WM + fm * 16 + coords[i] / 16, c = warp_n * WN + fn * 16 + coords[i] % 16;
                const int row = m0 + r, col = n0 + c;
                float out = 0.f;
                if (row < m && col < n) {
                    const float ws = wmode == WS_CHANNEL ? sb[col] : wmode == WS_TENSOR ? sb[0] : 1.f;
                    out = fmaf(accf[fm][fn][i], ws, bias ? to_float(bias[col]) : 0.f);
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

template<class T, int BM>
int launch_gemm_wmma(const uint8_t* a, const float* sa, const uint8_t* b, const float* sb, int wmode, const void* bias, const void* residual,
                     const void* rscale, void* y, int m, int n, int k, cudaStream_t stream) {
    const dim3 grid((n + WMMA_BN - 1) / WMMA_BN, (m + BM - 1) / BM);
    gemm_f8_wmma<T, BM><<<grid, 256, 0, stream>>>(a, sa, b, sb, wmode, static_cast<const T*>(bias), static_cast<const T*>(residual),
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

API int wf8_abi() { return 1; }

template<class T> int launch_quantize(const void* x, uint8_t* q, float* scale, int m, int k, int64_t ldx, int tile, cudaStream_t stream) {
    auto src = static_cast<const T*>(x);
    if (tile) {
        const dim3 grid((k + 127) / 128, (m + 127) / 128);
        quantize_tiles_f8<T><<<grid, 256, 0, stream>>>(src, q, scale, m, k, ldx);
    } else {
        const int chunks = (k + 255) / 256;
        quantize_rows_f8<T><<<unsigned((int64_t(m) * chunks + 7) / 8), 256, 0, stream>>>(src, q, scale, m, k, ldx, chunks);
    }
    return int(cudaGetLastError());
}

// E4M3 activations q (M, K) and FP32 scales (M, ceil(K / 128)) of x: dtype 0 = FP16, 1 = BF16, 2 = FP32; K a multiple of 8, rows 16-byte
// aligned (ldx in elements). tile 0: one scale per row and block (DeepGEMM per_token_cast_to_fp8); 1: one per 128 x 128 tile, stored for
// each of its rows (upstream's sm89 CUTLASS path). M at most 8388480 with tiles.
API int wf8_quantize(const void* x, uint8_t* q, float* scale, int m, int k, int64_t ldx, int dtype, int tile, cudaStream_t stream) {
    if (!m) return 0;
    if (k % 8) return -2;
    if (dtype == 0) return launch_quantize<__half>(x, q, scale, m, k, ldx, tile, stream);
    if (dtype == 1) return launch_quantize<__nv_bfloat16>(x, q, scale, m, k, ldx, tile, stream);
    if (dtype == 2) return launch_quantize<float>(x, q, scale, m, k, ldx, tile, stream);
    return -1;
}

#if !WGP_AMD
// Tile configurations: 0 = 128x128 (8 warps of 64x32), 1 = 64x128 (8 warps of 32x32), 2 = 128x128 with 128-element k-tiles (one per
// scale block, 3 stages), with E4M3 MMAs (SM89+); 10, 11 and 12 the same tiles with the FP16 emulation (SM80-88), 110 and 111 the Turing
// versions of 0 and 1 (3 stages: 64 KiB of shared memory per CTA).
template<int V> struct Variant;
template<> struct Variant<0> { static constexpr int BM = 128, BN = 128, BK = 64, WM = 64, WN = 32, STAGES = 4, CTAS = 1, MATH = MATH_E4M3; };
template<> struct Variant<1> { static constexpr int BM = 64, BN = 128, BK = 64, WM = 32, WN = 32, STAGES = 4, CTAS = 1, MATH = MATH_E4M3; };
template<> struct Variant<2> { static constexpr int BM = 128, BN = 128, BK = 128, WM = 64, WN = 32, STAGES = 3, CTAS = 1, MATH = MATH_E4M3; };
template<int V, int MATH_, int STAGES_> struct Derived : Variant<V> { static constexpr int MATH = MATH_, STAGES = STAGES_; };
template<> struct Variant<10> : Derived<0, MATH_F16, 4> {};
template<> struct Variant<11> : Derived<1, MATH_F16, 4> {};
template<> struct Variant<12> : Derived<2, MATH_F16, 3> {};
template<> struct Variant<110> : Derived<0, MATH_F16, 3> {};
template<> struct Variant<111> : Derived<1, MATH_F16, 3> {};

// Virtual architecture (major * 10 + minor) of the code the driver loaded for the current device, cached per device: the device's SASS,
// or the PTX it compiled for a newer GPU (or for this one, from a library built for older architectures only).
static int code_arch() {
    static int known[64] = {};
    int device = 0;
    cudaGetDevice(&device);
    if (device < 0 || device >= 64) device = 0;
    if (!known[device]) {
        cudaFuncAttributes attributes{};
        known[device] = cudaFuncGetAttributes(&attributes, code_arch_probe) == cudaSuccess ? attributes.ptxVersion : 89;
    }
    return known[device];
}

// The variant a configuration runs as with the loaded code: E4M3 MMAs from SM89, the FP16 emulation below, its Turing tiles below SM80.
// A library built for older architectures only (--arch 86, --arch 75) runs their kernels on any newer GPU (the emulated paths' tests).
static int device_variant(int variant) {
    const int arch = code_arch(), tile = variant % 10;
    if (tile > 2 || (variant != tile && variant != 10 + tile && variant != 110 + tile)) return -1;
    return arch >= 89 ? tile : arch >= 80 ? 10 + tile : 110 + tile % 2;  // Turing: 64-element k-tiles
}

API int wf8_device_variant(int variant) { return device_variant(variant); }
API int wf8_code_arch() { return code_arch(); }

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
    sizes[0] = s.sk_tiles ? int64_t(s.sk_ctas + s.sk_tiles) * V::BM * V::BN * 4 : 0;
    sizes[1] = 4 * int64_t(s.sk_tiles);
}

API void wf8_gemm_workspace(int m, int n, int k, int variant, int sms, int split, int64_t* sizes) {
    switch (device_variant(variant)) {
        case 0: return workspace_sizes<0>(m, n, k, sms, split, sizes);
        case 1: return workspace_sizes<1>(m, n, k, sms, split, sizes);
        case 2: return workspace_sizes<2>(m, n, k, sms, split, sizes);
        case 10: return workspace_sizes<10>(m, n, k, sms, split, sizes);
        case 11: return workspace_sizes<11>(m, n, k, sms, split, sizes);
        case 12: return workspace_sizes<12>(m, n, k, sms, split, sizes);
        case 110: return workspace_sizes<110>(m, n, k, sms, split, sizes);
        case 111: return workspace_sizes<111>(m, n, k, sms, split, sizes);
    }
    sizes[0] = sizes[1] = 0;
}

template<class T, int VI>
int launch_gemm(const uint8_t* a, const float* sa, const uint8_t* b, const float* sb, int wmode, const void* bias, const void* residual,
                const void* rscale, void* y, int m, int n, int k, int sms, int split, float* workspace, int* counters, cudaStream_t stream) {
    using V = Variant<VI>;
    using C = GemmConfig<V::BM, V::BN, V::BK, V::WM, V::WN, V::STAGES>;
    auto kernel = gemm_f8<T, V::BM, V::BN, V::BK, V::WM, V::WN, V::STAGES, V::CTAS, V::MATH>;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
        configured = true;
    }
    const Schedule s = make_schedule<V>(m, n, k, sms, split);
    kernel<<<s.grid, C::THREADS, C::SMEM, stream>>>(a, sa, b, sb, wmode, static_cast<const T*>(bias), static_cast<const T*>(residual),
                                                    static_cast<const T*>(rscale), static_cast<T*>(y), m, n, k, s, workspace, counters);
    return int(cudaGetLastError());
}

// Tile choice when untuned: estimated time = padded work / (relative mainloop efficiency * busy SMs). Stream-K keeps every SM busy
// once tiles x pieces (at most four pieces of four k-tiles) cover them.
template<int VI> double estimate(int m, int n, int k, int sms, double efficiency) {
    using V = Variant<VI>;
    sms *= V::CTAS;
    const long long tm = (m + V::BM - 1) / V::BM, tn = (n + V::BN - 1) / V::BN, kt = (k + V::BK - 1) / V::BK, tiles = tm * tn;
    long long pieces = kt / 4 < 4 ? kt / 4 : 4;
    if (pieces < 1) pieces = 1;
    const double busy = tiles >= sms ? sms : double(tiles * pieces < sms ? tiles * pieces : sms);
    return double(tm * V::BM) * double(tn * V::BN) * double(k) / (efficiency * busy);
}

API int wf8_gemm_select(int m, int n, int k, int sms) { return estimate<2>(m, n, k, sms, 1.0) <= estimate<1>(m, n, k, sms, 0.8) ? 2 : 1; }

template<class T> int dispatch_gemm(const uint8_t* a, const float* sa, const uint8_t* b, const float* sb, int wmode, const void* bias,
                                    const void* residual, const void* rscale, void* y, int m, int n, int k, int variant, int sms, int split,
                                    float* workspace, int* counters, cudaStream_t stream) {
    switch (device_variant(variant)) {
        case 0: return launch_gemm<T, 0>(a, sa, b, sb, wmode, bias, residual, rscale, y, m, n, k, sms, split, workspace, counters, stream);
        case 1: return launch_gemm<T, 1>(a, sa, b, sb, wmode, bias, residual, rscale, y, m, n, k, sms, split, workspace, counters, stream);
        case 2: return launch_gemm<T, 2>(a, sa, b, sb, wmode, bias, residual, rscale, y, m, n, k, sms, split, workspace, counters, stream);
        case 10: return launch_gemm<T, 10>(a, sa, b, sb, wmode, bias, residual, rscale, y, m, n, k, sms, split, workspace, counters, stream);
        case 11: return launch_gemm<T, 11>(a, sa, b, sb, wmode, bias, residual, rscale, y, m, n, k, sms, split, workspace, counters, stream);
        case 12: return launch_gemm<T, 12>(a, sa, b, sb, wmode, bias, residual, rscale, y, m, n, k, sms, split, workspace, counters, stream);
        case 110: return launch_gemm<T, 110>(a, sa, b, sb, wmode, bias, residual, rscale, y, m, n, k, sms, split, workspace, counters, stream);
        case 111: return launch_gemm<T, 111>(a, sa, b, sb, wmode, bias, residual, rscale, y, m, n, k, sms, split, workspace, counters, stream);
    }
    return -3;
}
#else
// AMD: whole tiles (split and the workspace are unused). Every configuration runs the 64x128 tile, whose block and promoted FP32
// accumulators fit the 256 registers (the 128-row tile spills).
API void wf8_gemm_workspace(int m, int n, int k, int variant, int sms, int split, int64_t* sizes) { sizes[0] = sizes[1] = 0; }
API int wf8_gemm_select(int m, int n, int k, int sms) { return 1; }
API int wf8_device_variant(int variant) { return variant % 10 <= 2 ? 1 : -1; }
API int wf8_code_arch() { return 0; }

template<class T> int dispatch_gemm(const uint8_t* a, const float* sa, const uint8_t* b, const float* sb, int wmode, const void* bias,
                                    const void* residual, const void* rscale, void* y, int m, int n, int k, int variant, int sms, int split,
                                    float* workspace, int* counters, cudaStream_t stream) {
    if (variant % 10 > 2) return -3;
    return launch_gemm_wmma<T, 64>(a, sa, b, sb, wmode, bias, residual, rscale, y, m, n, k, stream);
}
#endif

// Y (M, N) = sum over blocks of (A * sa) @ (B * sb).T + bias, or with a residual R (M, N): Y = R + rscale * (... + bias) (rscale: N values
// or null for 1), computed in FP32 and rounded once; Y may be R. A (M, K) and B (N, K) E4M3, sa (M, ceil(K / 128)) FP32; sb per wmode:
// 0 = (ceil(N / 128), ceil(K / 128)) blocks, 1 = (N) channels, 2 = (1) tensor, 3 = (N, ceil(K / 128)) channels and blocks of K. K a
// multiple of 16; A, B, R, Y contiguous. dtype of Y, bias, R, rscale: 0 = FP16, 1 = BF16. workspace / counters: wf8_gemm_workspace()
// sizes; counters zeroed, and left zeroed.
API int wf8_gemm(const uint8_t* a, const float* sa, const uint8_t* b, const float* sb, int wmode, const void* bias, const void* residual,
                 const void* rscale, void* y, int m, int n, int k, int dtype, int variant, int sms, int split, float* workspace, int* counters,
                 cudaStream_t stream) {
    if (!m || !n) return 0;
    if (k % 16 || wmode < WS_BLOCK || wmode > WS_GROUP) return -2;
    if (dtype == 0) return dispatch_gemm<__half>(a, sa, b, sb, wmode, bias, residual, rscale, y, m, n, k, variant, sms, split, workspace, counters, stream);
    if (dtype == 1) return dispatch_gemm<__nv_bfloat16>(a, sa, b, sb, wmode, bias, residual, rscale, y, m, n, k, variant, sms, split, workspace, counters, stream);
    return -1;
}

// ---------------------------------------------------------------------------------------------------------------------
// L2 eviction for timing: reading a buffer larger than the L2 cache evicts the weights, as in a model whose layers stream their
// weights from memory. The buffer comes from cudaMalloc, outside PyTorch's allocator, so that tuning during a CUDA graph's eager
// warmup cannot leave it in the graph's memory pool.
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
API int wf8_l2_flush(int device, int64_t bytes, cudaStream_t stream) {
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

API void wf8_l2_flush_release() {
    if (g_flush) (void)cudaFree(g_flush);
    g_flush = nullptr; g_flush_bytes = 0;
}
