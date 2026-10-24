# Copyright (c) 2026 DeepBeepMeep - Apache License 2.0 (see LICENSE.txt)
"""Block-sparse INT8-QK attention over 64-token tiles (WanGP VSA), built from attn.cu into wgp_attn.dll / libwgp_attn.so.

    from wgpkernels import attn
    attn.load_library()
    kind = attn.kernel_kind(device)        # "fp8" (sm_89, sm_120+), "fp16" (sm_80 / sm_86) or None
    attn.vsa_attention(q_int8, k_int8, v, out, lut, count, valid, q_scale, k_scale, v_scale, sm_scale)

q_int8, k_int8: int8 [S, H, 128] (S a multiple of 64), one FP32 scale per (head, 64-row block) in q_scale / k_scale [H, S // 64];
out: 16-bit [S, H, 128] (bf16 or fp16); lut int32 [H, S // 64, S // 64] holds, per query tile, the ascending key tile ids as increments
(first absolute, then differences) and count int32 [H, S // 64] the entries used; valid int32 [S // 64] is the real rows of every key
tile (a prefix). "fp8": v is fp8 e4m3 transposed [H, 128, S_pad] (S_pad >= S, a multiple of 64) with v_scale FP32 [H, 128], applied in
the epilogue (quantize v as v / v_scale with v_scale = max |v| / 2.25 per head channel). "fp16": v fp16 [S, H, 128], v_scale unused."""
import ctypes
from pathlib import Path

import torch

_lib = None


def default_library_path():
    import os
    return Path(__file__).with_name("wgp_attn.dll" if os.name == "nt" else "libwgp_attn.so")


def load_library(path=None):
    global _lib
    lib = ctypes.CDLL(str(Path(path or default_library_path()).resolve()))
    p, i, f = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
    lib.wgp_vsa_attn_fp8.argtypes = [p, p, p, p, p, p, p, p, p, p, i, i, i, i, i, f, p]
    lib.wgp_vsa_attn_fp16.argtypes = [p, p, p, p, p, p, p, p, p, i, i, i, f, p]
    lib.wgp_vsa_version.restype = i
    _lib = lib
    return lib


def kernel_kind(device):
    if torch.version.hip is not None:  # CUDA-only kernels (ROCm reports a compute capability too)
        return None
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) >= (8, 9):
        return "fp8"
    if major == 8:
        return "fp16"
    return None


def _check(code):
    if code:
        raise RuntimeError(f"WGPKernels VSA attention failed: {torch.cuda.CudartErrorString(code) if hasattr(torch.cuda, 'CudartErrorString') else code}")


def vsa_attention(q_int8, k_int8, v, out, lut, count, valid, q_scale, k_scale, v_scale, sm_scale, pv_fp16_accum=False):
    """out = block-sparse attention (see the module docstring); every tensor contiguous on one CUDA device. pv_fp16_accum (fp8 kernel):
    PV products accumulated in FP16 inside each block, SageAttention 2's faster variant, instead of FP32."""
    seq_len, heads, dim = q_int8.shape
    if dim != 128:
        raise ValueError("WGPKernels VSA attention supports head dim 128")
    for name, tensor in (("q", q_int8), ("k", k_int8), ("v", v), ("out", out), ("lut", lut), ("count", count), ("valid", valid), ("q_scale", q_scale), ("k_scale", k_scale)):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    stream = torch.cuda.current_stream(q_int8.device).cuda_stream
    out_bf16 = int(out.dtype == torch.bfloat16)
    if v.dtype == torch.float8_e4m3fn:
        _check(_lib.wgp_vsa_attn_fp8(q_int8.data_ptr(), k_int8.data_ptr(), v.data_ptr(), out.data_ptr(), lut.data_ptr(), count.data_ptr(), valid.data_ptr(),
                                     q_scale.data_ptr(), k_scale.data_ptr(), v_scale.data_ptr(), heads, seq_len, v.shape[-1], out_bf16, int(pv_fp16_accum),
                                     float(sm_scale), stream))
    else:
        _check(_lib.wgp_vsa_attn_fp16(q_int8.data_ptr(), k_int8.data_ptr(), v.data_ptr(), out.data_ptr(), lut.data_ptr(), count.data_ptr(), valid.data_ptr(),
                                      q_scale.data_ptr(), k_scale.data_ptr(), heads, seq_len, out_bf16, float(sm_scale), stream))
    return out
