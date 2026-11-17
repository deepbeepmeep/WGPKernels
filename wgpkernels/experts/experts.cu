// Copyright (c) 2026 DeepBeepMeep - Apache License 2.0 (see LICENSE.txt)
// With ideas of Strata: the fused int8 prompt experts of its src/prefill/moe_fused_iq.cu (MIT License, https://github.com/Niko1221/Strata, commit ffa80d7).
//
// Portions: Copyright (c) 2026 Niko1221 and the Strata contributors. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT
// WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// Routed experts (SwiGLU) of a pass of many tokens, on int8 tensor-core products, the weights left in their GGUF storage.
//
// The arithmetic. An activation block of 32 values is x = d_x * a, a = round(x / d_x) in -127..127, stored in natural order (per 64 values: 64 codes, {d_0, d_1} and 8 unused bytes). A weight block of 32 (or 16) values
// is w = d_w * q with q an int8: the i-quants' codebook entries with their signs applied, Q2_0's code - 1. So sum w x = d_w * d_x * sum q a with |sum q a| < 2^22, and the int32 dot becomes a float exactly with one float
// add: the product starts from the bits of 1.5 * 2^23, so as_float(D) - 1.5 * 2^23 is the dot. Per output and 32 values: that add, d_w * d_x, one fma (IQ2_S, a scale per 16 values: two dots, each times its d_w, then d_x).
//
// The load stage. Per 64 values of K (a stage), a thread decodes one 32-value sub-block of one of the work item's weight rows: its bytes are read with 16-bit loads (GGUF blocks are 2-byte aligned) one stage ahead into
// registers, two 256-value blocks ahead into L2, then turned into 32 int8 and the scales in shared memory, double-buffered. The codebooks live in shared memory. The activations arrive by cp.async, four stages deep; the
// fragments come by ldmatrix; the products are mma.sync m16n8k32 (m16n8k16 for IQ2_S).
//
// A work item is (64 ww weight rows) x (256 / ww routed rows of one expert, a tile); a persistent grid walks the tiles of a launch. Gate/up: gate and up rows of the item's features against the rows' tokens' activations,
// SwiGLU, H to int8 per 32 features, by row. Down: output rows against H, each output times its pair's probability, float32 by pair.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stddef.h>
#include <stdint.h>
#include "ggml_grids.h"

namespace {

constexpr int AB = 80;                  // activation bytes per 64 values
constexpr int MAGIC = 0x4B400000;       // the bits of 1.5 * 2^23
constexpr float MAGICF = 12582912.0f;
constexpr int THREADS = 512;            // 16 warps, each 64 weight rows x 16 routed rows: WW of them along the weight rows, 16 / WW along the routed rows
constexpr int WLD = 80;                 // bytes per decoded weight row of a stage: 64 int8 + 16 (no bank conflicts)
constexpr int ASTAGES = 4;              // cp.async depth of the activations
constexpr int T_IQ3_XXS = 18, T_IQ4_NL = 20, T_IQ3_S = 21, T_IQ2_S = 22, T_IQ4_XS = 23, T_Q2_0 = 42;  // ggml types

__host__ __device__ constexpr int tile_rows(int ww) { return THREADS / 32 / ww * 16; }
__host__ __device__ constexpr int weight_rows(int ww) { return 64 * ww; }
__host__ __device__ constexpr int block_bytes(int t) { return t == T_IQ2_S ? 82 : t == T_IQ3_XXS ? 98 : t == T_IQ3_S ? 110 : t == T_IQ4_XS ? 136 : 18; }
__host__ __device__ constexpr int grid_bytes(int t) { return t == T_IQ2_S ? 1024 * 8 : t == T_IQ3_XXS ? 256 * 4 : t == T_IQ3_S ? 512 * 4 : 0; }
__host__ __device__ constexpr size_t smem_bytes(int t, bool gu, int ww) {  // decoded weights and scales, activations, the rows of a tile, the codebook
    return (size_t) 2 * weight_rows(ww) * (WLD + 16) + (size_t) ASTAGES * tile_rows(ww) * AB + (size_t) tile_rows(ww) * (gu ? 4 : 8) + grid_bytes(t);
}

struct Args {
    const int64_t* table;     // [experts][3]: the addresses of the gate, up and down rows of an expert
    const int2* tiles;        // {expert | rows << 16, first row}, in the order of processing
    const int32_t* pairs;     // [rows]: the pair (token * topk + slot) of a row
    const float* probs;       // [pairs]: down only
    const uint8_t* act;       // gate/up: the tokens' activations; down: H by row
    uint8_t* mid;             // gate/up: H by row
    float* parts;             // down: float32 [pairs][hidden]
    int first, count;         // tiles of the launch
    int hidden, intermediate, topk;
    int row_bytes;            // of a weight row of the kernel's format
};

#if defined(__CUDA_ARCH__)
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(dst)), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void pf_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];\n" ::"l"(p)); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
// d = MAGIC + A (16 x 32 s8, row) * B (32 x 8 s8, col)
__device__ __forceinline__ void mma32(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(MAGIC));
}
// the same at m16n8k16 (A 16 x 16, B 16 x 8)
__device__ __forceinline__ void mma16(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(MAGIC));
}
__device__ __forceinline__ float dotf(int d) { return __int_as_float(d) - MAGICF; }
// four 8 x 16-byte matrices, row addresses from lanes 0-7, 8-15, 16-23, 24-31: lane (g, t) gets bytes 4t..4t+3 of row g of each, an mma fragment
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((unsigned) __cvta_generic_to_shared(p)) : "memory");
}
__device__ __forceinline__ uint32_t ld16(const uint8_t* p) { return *(const uint16_t*) p; }
__device__ __forceinline__ uint32_t ld32(const uint8_t* p) { return ld16(p) | (ld16(p + 2) << 16); }
__device__ __forceinline__ float half_at(uint32_t w) { return __half2float(__ushort_as_half((unsigned short) w)); }
// 7 bits of signs, the 8th their parity (bit 7 of v may be anything)
__device__ __forceinline__ uint32_t unpack_ksigns(uint32_t v) {
    v &= 0xFF;
    const uint32_t p = __popc(v) & 1;
    return (v ^ p << 7) * 0x01010101u;
}
// 8 bytes of a codebook entry with the sign byte s (broadcast) applied: bits 0-3 to qx, 4-7 to qy
__device__ __forceinline__ void signed8(uint32_t gx, uint32_t gy, uint32_t s, uint32_t& qx, uint32_t& qy) {
    const uint32_t m0 = __vcmpne4(s & 0x08040201u, 0), m1 = __vcmpne4(s & 0x80402010u, 0);
    qx = __vsub4(gx ^ m0, m0);
    qy = __vsub4(gy ^ m1, m1);
}
// 8 nibbles through a 16-entry int8 table (4 words): the low nibbles' values in lo, the high ones' in hi
__device__ __forceinline__ void table16(uint32_t q4, const uint32_t (&t)[4], uint32_t& lo, uint32_t& hi) {
    uint32_t tmp[2];
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t sh = 16 * i;
        const uint32_t l = __byte_perm(t[0], t[1], q4 >> sh), h = __byte_perm(t[2], t[3], q4 >> sh);
        tmp[i] = __byte_perm(l, h, sel >> sh);
    }
    lo = __byte_perm(tmp[0], tmp[1], 0x6420);
    hi = __byte_perm(tmp[0], tmp[1], 0x7531);
}
// Raw bytes of sub-block ib of the block at bp (i-quants: the 256-value block, ib 0..7; Q2_0: the 64-value block, ib 0..1; IQ4_NL: the 32-value block).
template <int T> __device__ __forceinline__ void load_unit(const uint8_t* bp, int ib, uint32_t (&w)[5]) {
    if constexpr (T == T_IQ2_S) {
        w[0] = ld32(bp + 2 + 4 * ib); w[1] = ld32(bp + 34 + 4 * ib);
        w[2] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16) | ((uint32_t) bp[74 + ib] << 24);
    } else if constexpr (T == T_IQ3_XXS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld32(bp + 66 + 4 * ib); w[3] = ld16(bp);
    } else if constexpr (T == T_IQ3_S) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld32(bp + 74 + 4 * ib);
        w[3] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16) | ((uint32_t) ((bp[106 + ib / 2] >> (4 * (ib & 1))) & 15) << 24);
    } else if constexpr (T == T_IQ4_XS) {
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = ld32(bp + 8 + 16 * ib + 4 * k);
        const uint32_t ls = ((bp[4 + ib / 2] >> (4 * (ib & 1))) & 15) | (((ld16(bp + 2) >> (2 * ib)) & 3) << 4);
        w[4] = ld16(bp) | (ls << 16);
    } else if constexpr (T == T_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = ld32(bp + 2 + 4 * k);
        w[4] = ld16(bp);
    } else {  // Q2_0
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp);
    }
}
// The sub-block as 32 int8 (q[0..7], natural order) and its scales (s0: values 0-15, s1: 16-31).
template <int T> __device__ __forceinline__ void convert(const uint32_t (&w)[5], const uint8_t* grid, const uint32_t (&kv)[4], uint32_t (&q)[8], float& s0, float& s1) {
    if constexpr (T == T_IQ2_S) {
        const uint2* g = (const uint2*) grid;
        const uint32_t qh = (w[2] >> 16) & 255;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 e = g[((w[0] >> (8 * l)) & 255) | ((qh << (8 - 2 * l)) & 0x300)];
            signed8(e.x, e.y, ((w[1] >> (8 * l)) & 255) * 0x01010101u, q[2 * l], q[2 * l + 1]);
        }
        const float d = half_at(w[2]);
        const uint32_t sc = w[2] >> 24;
        s0 = d * (float) (2 * (sc & 15) + 1) * 0.125f;
        s1 = d * (float) (2 * (sc >> 4) + 1) * 0.125f;
    } else if constexpr (T == T_IQ3_XXS) {
        const uint32_t* g = (const uint32_t*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t i0 = (w[l >> 1] >> (16 * (l & 1))) & 255, i1 = (w[l >> 1] >> (16 * (l & 1) + 8)) & 255;
            signed8(g[i0], g[i1], unpack_ksigns(w[2] >> (7 * l)), q[2 * l], q[2 * l + 1]);
        }
        s0 = s1 = half_at(w[3]) * (float) (2 * (w[2] >> 28) + 1) * 0.25f;
    } else if constexpr (T == T_IQ3_S) {
        const uint32_t* g = (const uint32_t*) grid;
        const uint32_t qh = (w[3] >> 16) & 255;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t i0 = (w[l >> 1] >> (16 * (l & 1))) & 255, i1 = (w[l >> 1] >> (16 * (l & 1) + 8)) & 255;
            signed8(g[i0 | ((qh << (8 - 2 * l)) & 256)], g[i1 | ((qh << (7 - 2 * l)) & 256)], ((w[2] >> (8 * l)) & 255) * 0x01010101u, q[2 * l], q[2 * l + 1]);
        }
        s0 = s1 = half_at(w[3]) * (float) (1 + 2 * (w[3] >> 24));
    } else if constexpr (T == T_IQ4_XS || T == T_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 4; ++k) table16(w[k], kv, q[k], q[4 + k]);
        s0 = s1 = T == T_IQ4_XS ? half_at(w[4]) * (float) ((int) (w[4] >> 16) - 32) : half_at(w[4]);
    } else {  // Q2_0: code - 1 through a byte table {-1, 0, 1, 2}
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            const uint32_t c = (w[h >> 1] >> (16 * (h & 1))) & 0xFFFF;
            const uint32_t qe = __byte_perm(0x020100FFu, 0x020100FFu, c & 0x7777);
            const uint32_t qo = __byte_perm(0x020100FFu, 0x020100FFu, (c >> 2) & 0x7777);
            q[2 * h] = __byte_perm(qe, qo, 0x5140);
            q[2 * h + 1] = __byte_perm(qe, qo, 0x7362);
        }
        s0 = s1 = half_at(w[2]);
    }
}
template <int T> __device__ __forceinline__ const void* grid_src() {
    if constexpr (T == T_IQ2_S) return iq2s_grid;
    else if constexpr (T == T_IQ3_XXS) return iq3xxs_grid;
    else if constexpr (T == T_IQ3_S) return iq3s_grid;
    else return nullptr;
}
#endif

__global__ void quantize_kernel(const uint16_t* __restrict__ x, int64_t chunks, uint8_t* __restrict__ codes) {
#if defined(__CUDA_ARCH__)
    const int64_t w = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;  // a warp per 64 values
    const int lane = threadIdx.x & 31;
    if (w >= chunks) return;
    const float v0 = __uint_as_float((uint32_t) x[w * 64 + lane] << 16), v1 = __uint_as_float((uint32_t) x[w * 64 + 32 + lane] << 16);  // BF16: the high half of a float
    float a0 = fabsf(v0), a1 = fabsf(v1);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, o));
        a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, o));
    }
    uint8_t* out = codes + w * AB;
    out[lane] = (uint8_t) (int8_t) (a0 > 0.0f ? __float2int_rn(v0 * (127.0f / a0)) : 0);
    out[32 + lane] = (uint8_t) (int8_t) (a1 > 0.0f ? __float2int_rn(v1 * (127.0f / a1)) : 0);
    if (lane == 0) *(float2*) (out + 64) = make_float2(a0 / 127.0f, a1 / 127.0f);
#endif
}

template <int WT, bool GU, int WW>
__global__ void __launch_bounds__(THREADS, 1) expert_kernel(const Args a) {
#if defined(__CUDA_ARCH__)
    constexpr int TR = tile_rows(WW), WR = weight_rows(WW);
    constexpr int WT_BYTES = WR * WLD, WS_FLOATS = WR * 4, ACT_STAGE = TR * AB;
    constexpr int BS = block_bytes(WT);
    constexpr bool K16 = WT == T_IQ2_S;
    const int NS = (GU ? a.hidden : a.intermediate) / 64;  // 64-value stages along K
    const int NFB = (GU ? 2 * a.intermediate : a.hidden) / WR;  // weight-row blocks per tile
    const int ACT_LD = NS * AB;  // bytes per activation row
    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t* wt = smem;  // [2][WR][WLD] decoded int8 weights
    float* ws = (float*) (smem + 2 * WT_BYTES);  // [2][WR][4] their scales per 16 values
    uint8_t* stages = (uint8_t*) (ws + 2 * WS_FLOATS);  // [ASTAGES][TR][AB] activations
    int* srow = (int*) (stages + ASTAGES * ACT_STAGE);  // gate/up: [TR] the token of each row of the tile; down: [TR] its pair, then [TR] its probability
    float* sprob = (float*) (srow + TR);
    uint8_t* sgrid = (uint8_t*) (srow + (GU ? TR : 2 * TR));

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, tig = lane & 3;
    const int wf = warp % WW, nb0 = 16 * (warp / WW);
    if constexpr (grid_bytes(WT) > 0) {
        const uint32_t* gs = (const uint32_t*) grid_src<WT>();
        for (int i = tid; i < grid_bytes(WT) / 4; i += THREADS) ((uint32_t*) sgrid)[i] = gs[i];
    }
    uint32_t kv[4] = {0, 0, 0, 0};
    if constexpr (WT == T_IQ4_XS || WT == T_IQ4_NL) {
        kv[0] = 0xBFAD9881u; kv[1] = 0xF6EADDCFu; kv[2] = 0x26190D01u; kv[3] = 0x71594535u;  // ggml's kvalues_iq4nl: -127, -104, -83, -65, ..., 113
    }
    // Local weight rows: m tile i of warp wf is rows 64 wf + 16 i.. (mma rows g and g + 8). For gate/up a tile is 8 features, its gate rows then its up rows, so a lane holds gate and up of one feature. The fragments come by
    // ldmatrix: lane l gives the address of row (l & 7) of matrix l >> 3 - A: {rows 0-7, rows 8-15} x {bytes 0-15, 16-31} of a 32-value chunk; B: {tokens nb0 + 0-7, + 8-15} x {bytes 0-15, 16-31}.
    const int lm = lane >> 3, l8 = lane & 7;
    const int a_off = (64 * wf + 8 * (lm & 1) + l8) * WLD + 16 * (lm >> 1);
    const int b_off = (nb0 + 8 * (lm >> 1) + l8) * AB + 16 * (lm & 1);
    // this thread's decode unit (the first 2 WR threads): local weight row ur, sub-block uj of each stage
    const bool dec = tid < 2 * WR;
    const int ur = dec ? tid >> 1 : 0, uj = tid & 1;
    const int nwork = a.count * NFB;
    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tile = a.tiles[a.first + w / NFB];
        const int fb = w % NFB, e = tile.x & 0xFFFF, nrows = tile.x >> 16, row0 = tile.y;
        const int rbase = fb * WR;
        // local row ur: gate/up - feature (WR / 2) fb + 8 (ur >> 4) + (ur & 7), its up row when bit 3 is set
        const uint8_t* wrow = dec ? (const uint8_t*) a.table[e * 3 + (GU ? (ur & 8 ? 1 : 0) : 2)] + (size_t) (GU ? fb * (WR / 2) + 8 * (ur >> 4) + (ur & 7) : rbase + ur) * a.row_bytes : nullptr;
        constexpr bool GU256 = GU && WT != T_Q2_0;  // gate/up in Q2_0 (64-value blocks): a block per stage, as the down rows read theirs
        auto unit = [&](int s) -> const uint8_t* {  // the block of stage s's sub-block
            if (GU256) return wrow + (s >> 2) * BS;
            return WT == T_IQ4_NL ? wrow + (2 * s + uj) * BS : wrow + s * BS;
        };
        auto sub = [&](int s) { return GU256 ? 2 * (s & 3) + uj : uj; };
        // the weight bytes into L2 ahead of the register loads: gate/up PF 256-value blocks ahead (the row's two threads take a block's first and last line), down the whole row slice at once
        constexpr int PF = 2;
        constexpr int SBB = WT == T_Q2_0 ? 4 * BS : BS;  // bytes per 256 values of a row
        const int sb_end = a.hidden / 256;
        auto prefetch_sb = [&](int sb) {
            if (dec && sb < sb_end) pf_l2(wrow + sb * SBB + (uj ? SBB - 1 : 0));
        };
        if (GU) {
#pragma unroll
            for (int sb = 0; sb < PF; ++sb) prefetch_sb(sb);
        } else if (dec) {
            for (int o = 128 * uj; o < a.row_bytes + 127; o += 256) pf_l2(wrow + min(o, a.row_bytes - 1));
        }
        __syncthreads();  // the previous item is done with the buffers
        if (tid < nrows) {
            const int pair = a.pairs[row0 + tid];
            if constexpr (GU) srow[tid] = pair / a.topk;
            else { srow[tid] = pair; sprob[tid] = a.probs[pair]; }
        }
        __syncthreads();  // srow
        // the rows past the item's end are not loaded: a warp entirely past it skips the products, and the columns of a partial one are never written (no reduction mixes columns)
        auto load_act = [&](int s) {
            uint8_t* st = stages + (s % ASTAGES) * ACT_STAGE;
            for (int c = tid; c < TR * 5; c += THREADS) {
                const int r = c / 5, q = c % 5;
                if (r < nrows) cp16(st + r * AB + q * 16, a.act + (size_t) (GU ? srow[r] : row0 + r) * ACT_LD + s * AB + q * 16);
            }
        };
        auto put = [&](const uint32_t (&raw)[5], int buf) {
            if (!dec) return;
            uint32_t q[8];
            float s0, s1;
            convert<WT>(raw, sgrid, kv, q, s0, s1);
            uint4* d = (uint4*) (wt + buf * WT_BYTES + ur * WLD + 32 * uj);
            d[0] = make_uint4(q[0], q[1], q[2], q[3]);
            d[1] = make_uint4(q[4], q[5], q[6], q[7]);
            *(float2*) (ws + buf * WS_FLOATS + ur * 4 + 2 * uj) = make_float2(s0, s1);
        };
#pragma unroll
        for (int s = 0; s < ASTAGES - 1; ++s) {
            if (s < NS) load_act(s);
            cp_commit();
        }
        uint32_t raw[5] = {0, 0, 0, 0, 0};
        if (dec) load_unit<WT>(unit(0), sub(0), raw);
        put(raw, 0);
        if (dec) load_unit<WT>(unit(1), sub(1), raw);
        // a warp whose rows are all past the tile's end only takes part in the loads
        const bool on0 = nb0 < nrows, on1 = nb0 + 8 < nrows;
        float acc[4][2][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int n = 0; n < 2; ++n)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[i][n][q] = 0.0f;
        for (int s = 0; s < NS; ++s) {
            cp_wait<ASTAGES - 2>();
            __syncthreads();
            if (s + ASTAGES - 1 < NS) load_act(s + ASTAGES - 1);
            cp_commit();
            if (GU && (s & 3) == 0) prefetch_sb((s >> 2) + PF);
            if (on0) {
                const uint8_t* W = wt + (s & 1) * WT_BYTES;
                const float* S = ws + (s & 1) * WS_FLOATS;
                const uint8_t* sa = stages + (s % ASTAGES) * ACT_STAGE;
                float2 dx[2][2];  // [n8 tile][column 2 tig + cc]: the activation scales per 32 values of the stage
#pragma unroll
                for (int n = 0; n < 2; ++n)
#pragma unroll
                    for (int cc = 0; cc < 2; ++cc) dx[n][cc] = *(const float2*) (sa + (nb0 + 8 * n + 2 * tig + cc) * AB + 64);
#pragma unroll
                for (int h = 0; h < 2; ++h) {  // the stage's two 32-value halves
                    uint32_t bq[4];  // {n0 bytes 0-15, n0 16-31, n1 0-15, n1 16-31}
                    ldsm4(bq, sa + b_off + 32 * h);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        uint32_t r[4];  // {rows g, g + 8} x {bytes 0-15, 16-31}
                        ldsm4(r, W + a_off + 16 * i * WLD + 32 * h);
                        const float2 swa = *(const float2*) (S + (64 * wf + 16 * i + g) * 4 + 2 * h);
                        const float2 swb = *(const float2*) (S + (64 * wf + 16 * i + 8 + g) * 4 + 2 * h);
#pragma unroll
                        for (int n = 0; n < 2; ++n) {
                            if (n == 1 && !on1) break;
                            if constexpr (K16) {  // a scale per 16 weights: the two halves' dots, each times its scale, then d_x
                                int d0[4], d1[4];
                                mma16(d0, r[0], r[1], bq[2 * n]);
                                mma16(d1, r[2], r[3], bq[2 * n + 1]);
#pragma unroll
                                for (int q = 0; q < 4; ++q) {
                                    const float v = fmaf(q < 2 ? swa.y : swb.y, dotf(d1[q]), (q < 2 ? swa.x : swb.x) * dotf(d0[q]));
                                    acc[i][n][q] = fmaf(h ? dx[n][q & 1].y : dx[n][q & 1].x, v, acc[i][n][q]);
                                }
                            } else {
                                int d[4];
                                mma32(d, r, bq[2 * n], bq[2 * n + 1]);
#pragma unroll
                                for (int q = 0; q < 4; ++q) {
                                    const float p = (q < 2 ? swa.x : swb.x) * (h ? dx[n][q & 1].y : dx[n][q & 1].x);
                                    acc[i][n][q] = fmaf(p, dotf(d[q]), acc[i][n][q]);
                                }
                            }
                        }
                    }
                }
            }
            // the next stage's weights (into the other buffer, read after the next barrier)
            if (s + 1 < NS) {
                put(raw, (s + 1) & 1);
                if (dec && s + 2 < NS) load_unit<WT>(unit(s + 2), sub(s + 2), raw);
            }
        }
        if (!on0) continue;
        if constexpr (GU) {
            // SwiGLU of feature (WR / 2) fb + 32 wf + 8 i + g (lane g of m tile i), tile rows nb0 + 8 n + 2 tig + c; H to int8 per row over the warp's 32 features (one 32-value half-block of the down product's K)
            const int blk = WW * fb + wf;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    float h[4], am = 0.0f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float gt = acc[i][n][c], up = acc[i][n][2 + c];
                        h[i] = gt / (1.0f + __expf(-gt)) * up;
                        am = fmaxf(am, fabsf(h[i]));
                    }
#pragma unroll
                    for (int o = 4; o < 32; o <<= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
                    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        uint8_t* o = a.mid + (size_t) (row0 + r) * (a.intermediate / 64 * AB) + (blk >> 1) * AB;
                        const int hh = blk & 1;
#pragma unroll
                        for (int i = 0; i < 4; ++i) o[32 * hh + 8 * i + g] = (uint8_t) (int8_t) __float2int_rn(h[i] * inv);
                        if (g == 0) *(float*) (o + 64 + 4 * hh) = am / 127.0f;
                    }
                }
            }
        } else {
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        float* row = a.parts + (size_t) srow[r] * a.hidden + rbase + 64 * wf + g;
                        const float p = sprob[r];
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            row[16 * i] = acc[i][n][c] * p;
                            row[16 * i + 8] = acc[i][n][2 + c] * p;
                        }
                    }
                }
            }
        }
    }
#endif
}

template <int T, bool GU>
int launch(int ww, int blocks, const Args& a, cudaStream_t stream) {
    const int items = a.count * ((GU ? 2 * a.intermediate : a.hidden) / weight_rows(ww));
    const unsigned grid = (unsigned) (items < blocks ? items : blocks);
    if (ww == 4) expert_kernel<T, GU, 4><<<grid, THREADS, smem_bytes(T, GU, 4), stream>>>(a);
    else expert_kernel<T, GU, 2><<<grid, THREADS, smem_bytes(T, GU, 2), stream>>>(a);
    return int(cudaGetLastError());
}

template <int T, bool GU, int WW>
int occupancy() {
    auto kernel = expert_kernel<T, GU, WW>;
    const size_t bytes = smem_bytes(T, GU, WW);
    int blocks = 0;
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) bytes) != cudaSuccess || cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, THREADS, bytes) != cudaSuccess) {
        (void) cudaGetLastError();
        return 0;
    }
    return blocks;
}

template <int T, bool GU> int occupancy_of(int ww) { return ww == 4 ? occupancy<T, GU, 4>() : occupancy<T, GU, 2>(); }

}  // namespace

#if defined(_WIN32)
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

API int wgx_abi() { return 1; }

// Prepares the kernel and returns how many of its blocks fit a multiprocessor (0: it does not run on this device).
API int wgx_blocks_per_sm(int device, int type, int gu, int ww) {
    if ((ww != 2 && ww != 4) || cudaSetDevice(device) != cudaSuccess) return 0;
    if (gu) switch (type) {
        case T_IQ2_S: return occupancy_of<T_IQ2_S, true>(ww);
        case T_IQ3_XXS: return occupancy_of<T_IQ3_XXS, true>(ww);
        case T_IQ3_S: return occupancy_of<T_IQ3_S, true>(ww);
        case T_IQ4_XS: return occupancy_of<T_IQ4_XS, true>(ww);
        case T_Q2_0: return occupancy_of<T_Q2_0, true>(ww);
    } else switch (type) {
        case T_IQ4_NL: return occupancy_of<T_IQ4_NL, false>(ww);
        case T_Q2_0: return occupancy_of<T_Q2_0, false>(ww);
    }
    return 0;
}

// x: BF16 [tokens][hidden], contiguous -> codes: int8 per 32 values, tokens * hidden / 64 * 80 bytes.
API int wgx_quantize(const void* x, int64_t tokens, int hidden, void* codes, cudaStream_t stream) {
    const int64_t chunks = tokens * (hidden / 64);
    quantize_kernel<<<(unsigned) ((chunks + 7) / 8), 256, 0, stream>>>((const uint16_t*) x, chunks, (uint8_t*) codes);
    return int(cudaGetLastError());
}

// The tiles [first, first + count): codes -> mid, H as int8 by row, rows * intermediate / 64 * 80 bytes.
API int wgx_gate_up(int type, int ww, int blocks, const void* table, const void* tiles, const void* pairs, const void* codes, void* mid, int first, int count, int hidden, int intermediate, int topk, int row_bytes, cudaStream_t stream) {
    const Args a{(const int64_t*) table, (const int2*) tiles, (const int32_t*) pairs, nullptr, (const uint8_t*) codes, (uint8_t*) mid, nullptr, first, count, hidden, intermediate, topk, row_bytes};
    switch (type) {
        case T_IQ2_S: return launch<T_IQ2_S, true>(ww, blocks, a, stream);
        case T_IQ3_XXS: return launch<T_IQ3_XXS, true>(ww, blocks, a, stream);
        case T_IQ3_S: return launch<T_IQ3_S, true>(ww, blocks, a, stream);
        case T_IQ4_XS: return launch<T_IQ4_XS, true>(ww, blocks, a, stream);
        case T_Q2_0: return launch<T_Q2_0, true>(ww, blocks, a, stream);
    }
    return -1;
}

// The tiles [first, first + count): mid -> parts, float32 [pairs][hidden]: each output of a pair times its probability.
API int wgx_down(int type, int ww, int blocks, const void* table, const void* tiles, const void* pairs, const void* probabilities, const void* mid, void* parts, int first, int count, int hidden, int intermediate, int row_bytes, cudaStream_t stream) {
    const Args a{(const int64_t*) table, (const int2*) tiles, (const int32_t*) pairs, (const float*) probabilities, (const uint8_t*) mid, nullptr, (float*) parts, first, count, hidden, intermediate, 1, row_bytes};
    switch (type) {
        case T_IQ4_NL: return launch<T_IQ4_NL, false>(ww, blocks, a, stream);
        case T_Q2_0: return launch<T_Q2_0, false>(ww, blocks, a, stream);
    }
    return -1;
}

// ---------------------------------------------------------------------------------------------------------------------
// The experts of a decoding window (with ideas of Strata: its native_expert_grouped, native_gu_swiglu_kernel and group_resident_kernel, src/kernels/cuda/iq_kernels.cu and s2_expert_grouped.cu, MIT License).
// Everything replays in a CUDA graph with any routing. A first launch quantizes the activations and groups the pairs by expert on the GPU, from ids and the pointer table, leaving out the experts without a copy. Each expert's
// rows are then decoded once for the tokens that read it (at most W_NC in one pass): with every expert in VRAM and rows read cold, a call is bound by the memory's bandwidth, so that what counts is reading fewer bytes.
// Activations and the SwiGLU's output: two int8 passes per value, x = d (hi + lo / 254) per 32 values, |error| <= block maximum / 64516; each dot is an exact int32, the first thing rounded being its conversion to float32.
// The codebooks are in shared memory: from global memory their lookups, random per lane, take most of the time of a call (T = 5: 107 us against 45).
// Gate/up: a block owns 32 features of a group of experts, 16 lanes a row, one thread reading a gate row and its up row; the SwiGLU is formed in the block. Down: a block owns 8 output rows, a warp each, walks the groups in their
// fixed order and adds to one accumulator per token (a lane), so that out is written directly, every row of every token, with no scratch per pair and no further launch; the next group's row is on its way to L2 meanwhile.
// ---------------------------------------------------------------------------------------------------------------------
namespace {

constexpr int W_NC = 4;  // entries of a group in one pass of its rows
constexpr int W_PAIRS = 128;  // tokens * topk at most: threads of the grouping block
constexpr int W_BLOCK = 80;  // bytes per 32 activations: 32 hi, 32 lo, their scale, unused
constexpr int W_ROWS = 8;  // output rows of a down block, a warp each: at most 32 tokens

struct WindowLayout { size_t starts, tokens, probs, rows, codes, mid, bytes; };

__host__ WindowLayout window_layout(int tokens, int topk, int hidden, int intermediate) {
    const size_t pairs = (size_t) tokens * topk;
    WindowLayout l;
    l.starts = 16;  // the counts before it: groups, entries
    l.tokens = l.starts + (pairs + 1) * 4;
    l.probs = l.tokens + pairs * 4;
    l.rows = (l.probs + pairs * 4 + 15) / 16 * 16;
    l.codes = l.rows + pairs * 24;
    l.mid = l.codes + (size_t) tokens * (hidden / 32) * W_BLOCK;
    l.bytes = l.mid + pairs * (intermediate / 32) * W_BLOCK;
    return l;
}

struct PlanArgs {
    const uint8_t* ids; const uint8_t* probabilities; const int64_t* table; const uint8_t* x;
    int32_t* counts; int32_t* starts; int32_t* tokens; float* probs; int64_t* rows; uint8_t* codes;
    int id_bytes, prob_bf16, x_fp32, pairs, topk, chunks;
};

struct WindowArgs {
    const int32_t* counts; const int32_t* starts; const int32_t* tokens; const float* probs; const int64_t* rows;  // the plan
    const uint8_t* act; uint8_t* mid; float* out;
    int n_tokens, hidden, intermediate, row_bytes;
};

#if defined(__CUDA_ARCH__)
__device__ __forceinline__ int dp(uint32_t a, uint32_t b, int c) { return __dp4a((int) a, (int) b, c); }  // signed bytes
__device__ __forceinline__ int dp4_low(const uint32_t (&q)[8], const uint4& x) { return dp(q[0], x.x, dp(q[1], x.y, dp(q[2], x.z, dp(q[3], x.w, 0)))); }
__device__ __forceinline__ int dp4_high(const uint32_t (&q)[8], const uint4& x, int s = 0) { return dp(q[4], x.x, dp(q[5], x.y, dp(q[6], x.z, dp(q[7], x.w, s)))); }
__device__ __forceinline__ int dp8(const uint32_t (&q)[8], const uint4& x0, const uint4& x1) { return dp4_high(q, x1, dp4_low(q, x0)); }

// acc + the 32 weights (q, scales s0 and s1 per 16 values) times the 32 activations of the block at p
template <bool K16> __device__ __forceinline__ float apply_block(const uint32_t (&q)[8], float s0, float s1, const uint8_t* p, float acc) {
    const uint4 h0 = *(const uint4*) p, h1 = *(const uint4*) (p + 16), l0 = *(const uint4*) (p + 32), l1 = *(const uint4*) (p + 48);
    const float dx = *(const float*) (p + 64);
    if constexpr (K16) {
        const int a = dp4_low(q, h0) * 254 + dp4_low(q, l0), b = dp4_high(q, h1) * 254 + dp4_high(q, l1);
        return fmaf(dx, fmaf(s1, (float) b, s0 * (float) a), acc);
    } else {
        return fmaf(s0 * dx, (float) (dp8(q, h0, h1) * 254 + dp8(q, l0, l1)), acc);
    }
}

__device__ __forceinline__ void q_store(const float v, uint8_t* o, int lane) {
    float m = fabsf(v);
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, d));
    const float inv = m > 1.0e-30f ? 127.0f / m : 0.0f;
    const float t = v * inv;
    const int hi = __float2int_rn(t);
    o[lane] = (uint8_t) (int8_t) hi;
    o[32 + lane] = (uint8_t) (int8_t) __float2int_rn((t - (float) hi) * 254.0f);
    if (lane == 0) *(float*) (o + 64) = m / (127.0f * 254.0f);
}

template <int WT, int N>
__device__ __forceinline__ void gu_pass(const WindowArgs& a, const uint8_t* ra, const uint8_t* rb, int e, int t, const uint8_t* grid, const uint32_t (&kv)[4], float (&sg)[N], float (&su)[N]) {
    constexpr bool Q2 = WT == T_Q2_0, K16 = WT == T_IQ2_S;
    const int act_row = (a.hidden / 32) * W_BLOCK, dots = a.hidden / 32;
    const uint8_t* xp[N];
#pragma unroll
    for (int c = 0; c < N; ++c) {
        xp[c] = a.act + (size_t) a.tokens[e + c] * act_row;
        sg[c] = 0.0f;
        su[c] = 0.0f;
    }
    for (int k = t; k < dots; k += 16) {
        const int offset = Q2 ? (k >> 1) * 18 : (k >> 3) * block_bytes(WT), ib = Q2 ? k & 1 : k & 7;
        uint32_t wa[5], wb[5], qa[8], qb[8];
        float a0, a1, b0, b1;
        load_unit<WT>(ra + offset, ib, wa);
        load_unit<WT>(rb + offset, ib, wb);
        convert<WT>(wa, grid, kv, qa, a0, a1);
        convert<WT>(wb, grid, kv, qb, b0, b1);
#pragma unroll
        for (int c = 0; c < N; ++c) {
            const uint8_t* p = xp[c] + k * W_BLOCK;
            sg[c] = apply_block<K16>(qa, a0, a1, p, sg[c]);
            su[c] = apply_block<K16>(qb, b0, b1, p, su[c]);
        }
    }
#pragma unroll
    for (int c = 0; c < N; ++c)
#pragma unroll
        for (int d = 8; d > 0; d >>= 1) {
            sg[c] += __shfl_xor_sync(0xffffffffu, sg[c], d);
            su[c] += __shfl_xor_sync(0xffffffffu, su[c], d);
        }
}

template <int WT, int N>
__device__ __forceinline__ void down_pass(const WindowArgs& a, const uint8_t* row, int e, int t, const uint32_t (&kv)[4], float (&s)[N]) {
    const int dots = a.intermediate / 32;
    const size_t h_row = (size_t) dots * W_BLOCK;
#pragma unroll
    for (int c = 0; c < N; ++c) s[c] = 0.0f;
    for (int k = t; k < dots; k += 32) {
        const int offset = WT == T_IQ4_NL ? k * 18 : (k >> 1) * 18;
        uint32_t w[5], q[8];
        float s0, s1;
        load_unit<WT>(row + offset, k & 1, w);
        convert<WT>(w, nullptr, kv, q, s0, s1);
#pragma unroll
        for (int c = 0; c < N; ++c) s[c] = apply_block<false>(q, s0, s1, a.mid + (size_t) (e + c) * h_row + k * W_BLOCK, s[c]);
    }
#pragma unroll
    for (int c = 0; c < N; ++c)
#pragma unroll
        for (int d = 16; d > 0; d >>= 1) s[c] += __shfl_xor_sync(0xffffffffu, s[c], d);
}
#endif

#if defined(__CUDA_ARCH__)
// exclusive prefix sum over the threads of the block (a multiple of 32); total: the block's sum
__device__ __forceinline__ int block_excl_sum(int v, int& total, int* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, warps = blockDim.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) sh[w] = x;
    __syncthreads();
    if (w == 0) {
        int s = lane < warps ? sh[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += y;
        }
        sh[lane] = s;
    }
    __syncthreads();
    const int r = x - v + (w > 0 ? sh[w - 1] : 0);
    total = sh[warps - 1];
    __syncthreads();  // sh is reused by the next call
    return r;
}
#endif

// One block for the plan, the others for the activations of the window, a warp per 32 values. Groups in the order of first appearance; the order of the entries of a group changes no value.
__global__ void __launch_bounds__(W_PAIRS) window_plan_kernel(const PlanArgs a) {
#if defined(__CUDA_ARCH__)
    extern __shared__ int s_expert[];  // per expert: its first pair, its pairs
    __shared__ int s_sum[32], s_begin[W_PAIRS];
    const int i = threadIdx.x;
    if (blockIdx.x > 0) {
        const int64_t w = (int64_t) (blockIdx.x - 1) * 4 + (i >> 5);
        const int lane = i & 31;
        if (w >= a.chunks) return;
        const float v = a.x_fp32 ? ((const float*) a.x)[w * 32 + lane] : __uint_as_float((uint32_t) ((const uint16_t*) a.x)[w * 32 + lane] << 16);  // BF16: the high half of a float
        q_store(v, a.codes + w * W_BLOCK, lane);
        return;
    }
    int expert = -1, rank = 0, first = i, size = 0;
    int64_t rows[3] = {0, 0, 0};
    if (i < a.pairs) {
        const int id = *(const int32_t*) (a.ids + (size_t) i * a.id_bytes);
        rows[0] = a.table[(size_t) id * 3]; rows[1] = a.table[(size_t) id * 3 + 1]; rows[2] = a.table[(size_t) id * 3 + 2];
        if (rows[0] != 0 && rows[1] != 0 && rows[2] != 0) {  // no address of a pair without a whole row of the table is ever read
            expert = id;
            s_expert[id * 2] = W_PAIRS;
            s_expert[id * 2 + 1] = 0;
        }
    }
    __syncthreads();
    if (expert >= 0) {
        atomicMin(&s_expert[expert * 2], i);
        rank = atomicAdd(&s_expert[expert * 2 + 1], 1);
    }
    __syncthreads();
    if (expert >= 0) {
        first = s_expert[expert * 2];
        if (first == i) size = s_expert[expert * 2 + 1];
    }
    int groups, entries;
    const int group = block_excl_sum(size > 0 ? 1 : 0, groups, s_sum), begin = block_excl_sum(size, entries, s_sum);
    if (size > 0) s_begin[i] = begin;
    if (i == 0) {
        a.counts[0] = groups;
        a.counts[1] = entries;
        a.starts[groups] = entries;
    }
    __syncthreads();
    if (expert >= 0) {
        const int at = s_begin[first] + rank;
        a.tokens[at] = i / a.topk;
        a.probs[at] = a.prob_bf16 ? __uint_as_float((uint32_t) ((const uint16_t*) a.probabilities)[i] << 16) : ((const float*) a.probabilities)[i];
        if (size > 0) {
            a.starts[group] = begin;
            a.rows[(size_t) group * 3] = rows[0]; a.rows[(size_t) group * 3 + 1] = rows[1]; a.rows[(size_t) group * 3 + 2] = rows[2];
        }
    }
#endif
}

// A block owns 32 features of a group of experts (grid: features / 32, a block row per possible group: those past the window's groups return at once): 32 sub-warps of 16 lanes, each one computing a gate row and its up row for the group's entries.
template <int WT>
__global__ void __launch_bounds__(THREADS) window_gu_kernel(const WindowArgs a) {
#if defined(__CUDA_ARCH__)
    const int groups = *a.counts, g = blockIdx.y;
    if (g >= groups) return;
    constexpr bool STAGED = grid_bytes(WT) > 0;
    __shared__ __align__(16) uint32_t s_grid[STAGED ? grid_bytes(WT) / 4 : 4];
    __shared__ float sh[W_NC][32];
    const int tid = threadIdx.x, sw = tid >> 4, t = tid & 15, warp = tid >> 5, lane = tid & 31;
    if constexpr (STAGED) {
        const uint32_t* gs = (const uint32_t*) grid_src<WT>();
        for (int i = tid; i < grid_bytes(WT) / 4; i += THREADS) s_grid[i] = gs[i];
        __syncthreads();
    }
    const uint8_t* grid = (const uint8_t*) s_grid;
    uint32_t kv[4] = {0, 0, 0, 0};
    if constexpr (WT == T_IQ4_XS) { kv[0] = 0xBFAD9881u; kv[1] = 0xF6EADDCFu; kv[2] = 0x26190D01u; kv[3] = 0x71594535u; }
    const size_t off = (size_t) (blockIdx.x * 32 + sw) * a.row_bytes;
    const uint8_t* ra = (const uint8_t*) a.rows[(size_t) g * 3] + off;
    const uint8_t* rb = (const uint8_t*) a.rows[(size_t) g * 3 + 1] + off;
    const int e0 = a.starts[g], e1 = a.starts[g + 1];
    for (int e = e0; e < e1; e += W_NC) {
        const int n = min(W_NC, e1 - e);
        float gv[W_NC], uv[W_NC];
        if (n == 1) {
            float g1[1], u1[1];
            gu_pass<WT, 1>(a, ra, rb, e, t, grid, kv, g1, u1);
            gv[0] = g1[0]; uv[0] = u1[0];
        } else if (n == 2) {
            float g2[2], u2[2];
            gu_pass<WT, 2>(a, ra, rb, e, t, grid, kv, g2, u2);
            gv[0] = g2[0]; uv[0] = u2[0]; gv[1] = g2[1]; uv[1] = u2[1];
        } else if (n == 3) {
            float g3[3], u3[3];
            gu_pass<WT, 3>(a, ra, rb, e, t, grid, kv, g3, u3);
            gv[0] = g3[0]; uv[0] = u3[0]; gv[1] = g3[1]; uv[1] = u3[1]; gv[2] = g3[2]; uv[2] = u3[2];
        } else {
            gu_pass<WT, 4>(a, ra, rb, e, t, grid, kv, gv, uv);
        }
#pragma unroll
        for (int c = 0; c < W_NC; ++c)
            if (c < n && t == c) sh[c][sw] = gv[c] / (1.0f + expf(-gv[c])) * uv[c];  // one lane of 16 per row, so the precise exp costs little
        __syncthreads();
        if (warp < n) q_store(sh[warp][lane], a.mid + (size_t) (e + warp) * (a.intermediate / 32) * W_BLOCK + blockIdx.x * W_BLOCK, lane);
        __syncthreads();
    }
#endif
}

// A block owns W_ROWS output rows, a warp each; lane t holds the running sum of token t.
template <int WT>
__global__ void __launch_bounds__(256) window_down_kernel(const WindowArgs a) {
#if defined(__CUDA_ARCH__)
    __shared__ int s_start[W_PAIRS + 1], s_token[W_PAIRS];
    __shared__ float s_prob[W_PAIRS];
    __shared__ __align__(8) int64_t s_row[W_PAIRS];
    const int groups = a.counts[0], entries = a.counts[1], tid = threadIdx.x, t = tid & 31;
    for (int i = tid; i <= groups; i += 256) {
        s_start[i] = a.starts[i];
        if (i < groups) s_row[i] = a.rows[(size_t) i * 3 + 2];
    }
    for (int i = tid; i < entries; i += 256) {
        s_token[i] = a.tokens[i];
        s_prob[i] = a.probs[i];
    }
    __syncthreads();
    uint32_t kv[4] = {0, 0, 0, 0};
    if constexpr (WT == T_IQ4_NL) { kv[0] = 0xBFAD9881u; kv[1] = 0xF6EADDCFu; kv[2] = 0x26190D01u; kv[3] = 0x71594535u; }
    const size_t off = (size_t) (blockIdx.x * W_ROWS + (tid >> 5)) * a.row_bytes;
    const bool prefetching = t * 64 < a.row_bytes;
    float total = 0.0f;
    if (prefetching && groups > 0) pf_l2((const uint8_t*) s_row[0] + off + t * 64);
    for (int g = 0; g < groups; ++g) {
        const uint8_t* row = (const uint8_t*) s_row[g] + off;
        const int e0 = s_start[g], e1 = s_start[g + 1];
        if (prefetching && g + 1 < groups) pf_l2((const uint8_t*) s_row[g + 1] + off + t * 64);  // the next group's row while this one is computed
        for (int e = e0; e < e1; e += W_NC) {
            const int n = min(W_NC, e1 - e);
            float s[W_NC];
            if (n == 1) {
                float s1[1];
                down_pass<WT, 1>(a, row, e, t, kv, s1);
                s[0] = s1[0];
            } else if (n == 2) {
                float s2[2];
                down_pass<WT, 2>(a, row, e, t, kv, s2);
                s[0] = s2[0]; s[1] = s2[1];
            } else if (n == 3) {
                float s3[3];
                down_pass<WT, 3>(a, row, e, t, kv, s3);
                s[0] = s3[0]; s[1] = s3[1]; s[2] = s3[2];
            } else {
                down_pass<WT, 4>(a, row, e, t, kv, s);
            }
#pragma unroll
            for (int c = 0; c < W_NC; ++c)
                if (c < n && s_token[e + c] == t) total = fmaf(s[c], s_prob[e + c], total);
        }
    }
    if (t < a.n_tokens) a.out[(size_t) t * a.hidden + blockIdx.x * W_ROWS + (tid >> 5)] = total;
#endif
}

template <int WT> int window_gu(int group_rows, const WindowArgs& a, cudaStream_t stream) {
    window_gu_kernel<WT><<<dim3((unsigned) (a.intermediate / 32), (unsigned) group_rows), THREADS, 0, stream>>>(a);
    return int(cudaGetLastError());
}

template <int WT> int window_down(const WindowArgs& a, cudaStream_t stream) {
    window_down_kernel<WT><<<(unsigned) (a.hidden / W_ROWS), 256, 0, stream>>>(a);
    return int(cudaGetLastError());
}

}  // namespace

API int64_t wgx_window_scratch(int tokens, int topk, int hidden, int intermediate) { return (int64_t) window_layout(tokens, topk, hidden, intermediate).bytes; }

// The plan (parts & 1), gate/up and SwiGLU (parts & 2), down into out (parts & 4), on the stream; no read on the host, grids from tokens, topk and the sizes only. out: float32 [tokens][hidden], every row written.
API int wgx_window_forward(int gate, int down, int parts, const void* x, int x_fp32, const void* ids, int id_bytes, const void* probabilities, int prob_bf16, const void* pointers, int experts, int tokens, int topk, int hidden, int intermediate, void* out, void* scratch, cudaStream_t stream) {
    const WindowLayout l = window_layout(tokens, topk, hidden, intermediate);
    uint8_t* base = (uint8_t*) scratch;
    int status = 0;
    if (parts & 1) {
        const int chunks = tokens * (hidden / 32);
        const PlanArgs p{(const uint8_t*) ids, (const uint8_t*) probabilities, (const int64_t*) pointers, (const uint8_t*) x, (int32_t*) base, (int32_t*) (base + l.starts), (int32_t*) (base + l.tokens), (float*) (base + l.probs), (int64_t*) (base + l.rows), base + l.codes, id_bytes, prob_bf16, x_fp32, tokens * topk, topk, chunks};
        window_plan_kernel<<<(unsigned) (1 + (chunks + 3) / 4), W_PAIRS, (size_t) experts * 8, stream>>>(p);
        status = int(cudaGetLastError());
    }
    WindowArgs a{(const int32_t*) base, (const int32_t*) (base + l.starts), (const int32_t*) (base + l.tokens), (const float*) (base + l.probs), (const int64_t*) (base + l.rows), base + l.codes, base + l.mid, (float*) out, tokens, hidden, intermediate, 0};
    if (!status && (parts & 2)) {
        const int rows = tokens * topk;
        a.row_bytes = gate == T_Q2_0 ? hidden / 64 * 18 : hidden / 256 * block_bytes(gate);
        switch (gate) {
            case T_IQ2_S: status = window_gu<T_IQ2_S>(rows, a, stream); break;
            case T_IQ3_XXS: status = window_gu<T_IQ3_XXS>(rows, a, stream); break;
            case T_IQ3_S: status = window_gu<T_IQ3_S>(rows, a, stream); break;
            case T_IQ4_XS: status = window_gu<T_IQ4_XS>(rows, a, stream); break;
            case T_Q2_0: status = window_gu<T_Q2_0>(rows, a, stream); break;
            default: return -1;
        }
    }
    if (!status && (parts & 4)) {
        a.row_bytes = down == T_IQ4_NL ? intermediate / 32 * 18 : intermediate / 64 * 18;
        if (down == T_IQ4_NL) status = window_down<T_IQ4_NL>(a, stream);
        else if (down == T_Q2_0) status = window_down<T_Q2_0>(a, stream);
        else return -1;
    }
    return status;
}
