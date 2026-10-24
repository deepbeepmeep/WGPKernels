// Copyright (c) 2026 DeepBeepMeep - Apache License 2.0 (see LICENSE.txt)
// Block-sparse INT8-QK attention over 64-token tiles for WanGP's VSA (video sparse attention): every query tile attends the key
// tiles of its index list, partial tiles masked by their valid-token count. Kernels derived from SpargeAttn / SageAttention 2
// (sparge/, Apache-2.0): FP8 PV with per-channel V scales on sm_89 and later, FP16 PV on sm_80/sm_86.
//
// Layouts (batch 1): q, k int8 [S, H, 128] with one FP32 scale per (head, 64-row block); o 16-bit [S, H, 128];
// sm_89: v fp8 e4m3 transposed [H, 128, S_pad] with FP32 scales [H, 128]; sm_80: v fp16 [S, H, 128].
// lut int32 [H, S / 64, S / 64]: per query tile the key tile ids as increments (first absolute, then differences, ascending);
// count int32 [H, S / 64]: entries used per query tile; valid int32 [S / 64]: real rows per key tile (a prefix of the tile).
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <type_traits>

#include "sparge/qattn/vsa_sm89.cuh"
#include "sparge/qattn/vsa_sm80.cuh"

#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

namespace {
constexpr uint32_t CTA_Q = 64, CTA_K = 64, WARP_Q = 16, WARP_K = 64, HEAD_DIM = 128;
constexpr uint32_t PER_BLOCK = 1;  // QuantGranularity::kPerBlock for q and k
}

// FP8 PV (sm_89, sm_120 and later; sm_100 too): accumulation in FP32, V scales applied in the epilogue.
// pv_fp16_accum: the PV products accumulate in FP16 inside a block (SageAttention 2's faster variant), else in FP32.
API int wgp_vsa_attn_fp8(const void* q, const void* k, const void* v, void* o, const int32_t* lut, const int32_t* count, const int32_t* valid,
                         const float* q_scale, const float* k_scale, const float* v_scale, int heads, int seq_len, int v_pad_len, int out_bf16,
                         int pv_fp16_accum, float sm_scale, void* stream) {
    if (seq_len % CTA_K != 0 || v_pad_len < seq_len) return cudaErrorInvalidValue;
    const uint32_t stride_seq = heads * HEAD_DIM, stride_h = HEAD_DIM;
    auto launch = [&](auto* out, auto accum) {
        using DTypeOut = std::remove_pointer_t<decltype(out)>;
        SpargeAttentionSM89Dispatched<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, PER_BLOCK, float, true, decltype(accum)::value, 0, DTypeOut, false, true, false>(
            (int8_t*)q, (int8_t*)k, (__nv_fp8_e4m3*)v, out, nullptr, (int32_t*)lut, (int32_t*)count, (int32_t*)valid, nullptr,
            (float*)q_scale, (float*)k_scale, (float*)v_scale, 1, seq_len, seq_len, heads, heads,
            0, stride_seq, stride_h, 0, stride_seq, stride_h, 0, HEAD_DIM * v_pad_len, v_pad_len, 0, stride_seq, stride_h, sm_scale,
            (cudaStream_t)stream);
    };
    if (pv_fp16_accum) {
        if (out_bf16) launch((nv_bfloat16*)o, std::true_type{}); else launch((half*)o, std::true_type{});
    } else {
        if (out_bf16) launch((nv_bfloat16*)o, std::false_type{}); else launch((half*)o, std::false_type{});
    }
    return cudaGetLastError();
}

// FP16 PV with FP16 accumulation (sm_80 / sm_86), v fp16 [S, H, 128].
API int wgp_vsa_attn_fp16(const void* q, const void* k, const void* v, void* o, const int32_t* lut, const int32_t* count, const int32_t* valid,
                          const float* q_scale, const float* k_scale, int heads, int seq_len, int out_bf16, float sm_scale, void* stream) {
    if (seq_len % CTA_K != 0) return cudaErrorInvalidValue;
    const uint32_t stride_seq = heads * HEAD_DIM, stride_h = HEAD_DIM;
    auto launch = [&](auto* out) {
        using DTypeOut = std::remove_pointer_t<decltype(out)>;
        SpargeAttentionSM80Dispatched<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, PER_BLOCK, half, true, 0, DTypeOut, false, false>(
            (int8_t*)q, (int8_t*)k, (half*)v, out, nullptr, (int32_t*)lut, (int32_t*)count, (int32_t*)valid, nullptr,
            (float*)q_scale, (float*)k_scale, 1, seq_len, seq_len, heads, heads,
            0, stride_seq, stride_h, 0, stride_seq, stride_h, 0, stride_seq, stride_h, 0, stride_seq, stride_h, sm_scale, (cudaStream_t)stream);
    };
    if (out_bf16) launch((nv_bfloat16*)o); else launch((half*)o);
    return cudaGetLastError();
}

API int wgp_vsa_version() { return 1; }
