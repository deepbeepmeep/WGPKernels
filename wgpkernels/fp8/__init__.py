# Copyright (c) 2026 DeepBeepMeep - Apache License 2.0 (see LICENSE.txt)
"""FP8 (E4M3) block-scaled linears for FP8 checkpoints, with the semantics of DeepGEMM's FP8 GEMM and CUTLASS's GemmBlockwise:
Y = sum over the 128-element blocks kb of K of sa[m, kb] * sb[n // 128, kb] * (Aq[m, kb] . Bq[n, kb]) + bias, each block's dot product
accumulated from zero in FP32 and promoted into FP32 accumulators. Kernels come from an ahead-of-time built library
(python -m wgpkernels.build --library fp8).

- Weights: the checkpoint's (N, K) E4M3 matrix (never dequantized or copied) with FP32 scales per 128 x 128 block (ceil(N / 128),
  ceil(K / 128)) as DeepGEMM's per_block_cast_to_fp8 makes them, or per output channel and 128-element block of K (N, ceil(K / 128)),
  both folded at every promotion, or per output channel (N) or tensor (1), applied in the epilogue.
- Activations: quantized by quantize_blocks() to E4M3 with one FP32 scale per row and 128-element block (amax / 448, DeepGEMM's
  per_token_cast_to_fp8), or tile=True: one scale per 128 x 128 tile, the activation scaling of upstream's sm89 CUTLASS path. Values
  round to nearest even as PyTorch's .to(torch.float8_e4m3fn) and saturate to +/-448.
- Tensor cores: E4M3 MMAs on SM89+ (RTX 40/50); on SM80-88 and Turing (SM75) the E4M3 bytes are decoded exactly to FP16 in registers
  (FP16 MMAs, FP32 accumulation: the same products, another summation order); AMD RDNA4: rocWMMA's gfx12 FP8 WMMA.
- Persistent tiles with stream-K for the last wave; split tiles add their FP32 partial sums in a fixed order: results are deterministic.

FP8 W8A8 is approximate: the activation quantization dominates the error (measured per layer: about 10x the error of the same FP8
weights with BF16 activations, 4x when activations have outlier channels, which block scales isolate). Fusions: bias, residual epilogue
Y = R + residual_scale * (X @ W.T + bias) computed in FP32 and rounded once (out may be R).
Tile configurations are timed per GPU, kernel path and shape bucket on the first call and cached in ~/.triton/autotune/.
Activations are quantized in row chunks bounded by SCRATCH_BYTES, so no full-size quantized copy is allocated.
"""
import ctypes
import functools
import json
import os
import time
from pathlib import Path

import torch

SCRATCH_BYTES = 16 * 1024 * 1024
WS_BLOCK, WS_CHANNEL, WS_TENSOR, WS_GROUP = 0, 1, 2, 3  # weight scale granularity
_COUNTERS = 1024  # split tiles never exceed the SM count
HIP_ARCHS = ("gfx1200", "gfx1201")  # the AMD targets of the HIP library (RDNA4)
_lib = None
_DTYPES = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}
_FP8 = (torch.float8_e4m3fn, torch.uint8, torch.int8)  # E4M3 bytes, or their raw views
_sms = {}
_counters = {}
_sizes = (ctypes.c_int64 * 2)()


def default_library_path():
    return Path(__file__).with_name("wgp_fp8.dll" if os.name == "nt" else "libwgp_fp8.so")


def load_library(path=None):
    global _lib
    lib = ctypes.CDLL(str(Path(path or default_library_path()).resolve()))
    if lib.wf8_abi() != 1:
        raise RuntimeError("FP8 GEMM library ABI mismatch; rebuild it")
    p, i, i64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64
    lib.wf8_quantize.argtypes = [p, p, p, i, i, i64, i, i, p]
    lib.wf8_gemm.argtypes = [p, p, p, p, i, p, p, p, p, i, i, i, i, i, i, i, p, p, p]
    lib.wf8_gemm_workspace.argtypes = [i, i, i, i, i, i, p]
    lib.wf8_gemm_workspace.restype = None
    lib.wf8_gemm_select.argtypes = [i, i, i, i]
    lib.wf8_device_variant.argtypes = [i]
    lib.wf8_l2_flush.argtypes = [i, i64, p]
    lib.wf8_l2_flush_release.restype = None
    for index in range(torch.cuda.device_count()):
        # ROCm: only the GPUs the library is built for (PyTorch has no kernels for others, e.g. a Ryzen integrated GPU beside the RDNA4 one)
        if torch.version.hip is None or torch.cuda.get_device_properties(index).gcnArchName.split(":")[0] in HIP_ARCHS:
            _counters.setdefault(index, torch.zeros(_COUNTERS, dtype=torch.int32, device=torch.device("cuda", index)))
    _lib = lib
    return lib


def _check(code):
    if code:
        raise RuntimeError(f"FP8 GEMM launch failed (CUDA error {code})")


def _ptr(tensor):
    return None if tensor is None else tensor.data_ptr()


def _stream(device):
    return torch._C._cuda_getCurrentRawStream(device.index)


def _sm_count(device):
    count = _sms.get(device.index)
    if count is None:
        count = _sms[device.index] = torch.cuda.get_device_properties(device).multi_processor_count
    return count


def _zeroed_counters(device, size):
    """Split-K tile counters, allocated when the library loads (never inside CUDA graph capture) and left zeroed by the kernels.
    Concurrent calls on several streams of one device would share them: use one stream per device."""
    if size > _COUNTERS * 4:
        raise RuntimeError("FP8 GEMM split-K counters exceeded")
    return _counters[device.index]


def kernel_path():
    """Kernels the current device runs: 'e4m3' (E4M3 tensor cores), 'fp16' (E4M3 decoded to FP16), 'sm75' (Turing) or 'wmma' (AMD).
    They follow the code the driver loads (code_arch()): a library built for older architectures only (--arch 86, --arch 75) runs their
    emulated kernels on a newer GPU, which is how those paths are tested on an SM89+ GPU."""
    if torch.version.hip is not None:
        return 'wmma'
    return {0: 'e4m3', 10: 'fp16', 110: 'sm75'}[_lib.wf8_device_variant(0)]


def code_arch():
    """Virtual architecture (major * 10 + minor) of the library code the driver loaded for the current device: its SASS, or the PTX it
    compiled (0 on AMD)."""
    return _lib.wf8_code_arch()


def quantize_blocks(x, tile=False):
    """E4M3 activations of x (2-D FP16/BF16/FP32, rows 16-byte aligned, unit column stride, K a multiple of 16) with one FP32 scale per
    row and 128-element block: (q (M, K) float8_e4m3fn, scales (M, ceil(K / 128))). scale = max(amax, 1e-4) / 448 and q = e4m3(x * (1 /
    scale)), as DeepGEMM's per_token_cast_to_fp8; tile=True: amax over each 128 x 128 tile and q = e4m3(x / scale), as upstream's sm89
    CUTLASS path, the tile's scale repeated on its rows."""
    m, k = x.shape
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((m, (k + 127) // 128), dtype=torch.float32, device=x.device)
    _check(_lib.wf8_quantize(x.data_ptr(), q.data_ptr(), scale.data_ptr(), m, k, x.stride(0), _DTYPES[x.dtype], int(tile), _stream(x.device)))
    return q, scale


def quantize_weight(w):
    """E4M3 weights with one FP32 scale per 128 x 128 block, as DeepGEMM's per_block_cast_to_fp8 (PyTorch, for checkpoints stored in
    BF16/FP16/FP32): (q (N, K) float8_e4m3fn, scales (ceil(N / 128), ceil(K / 128)))."""
    n, k = w.shape
    padded = torch.zeros((-(-n // 128) * 128, -(-k // 128) * 128), dtype=w.dtype, device=w.device)
    padded[:n, :k] = w
    view = padded.view(-1, 128, padded.shape[1] // 128, 128)
    scale = view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4) / 448.0
    q = (view * (1.0 / scale)).to(torch.float8_e4m3fn)
    return q.view_as(padded)[:n, :k].contiguous(), scale.view(view.shape[0], view.shape[2])


def _scale_mode(sb, n, k):
    """Weight scale granularity of sb: per 128 x 128 block, per tensor, per output channel, or per output channel and 128-element block of
    K (where shapes coincide, the semantics do too)."""
    kblocks = -(-k // 128)
    if sb.dim() == 2 and tuple(sb.shape) == (-(-n // 128), kblocks):
        return WS_BLOCK
    if sb.numel() == 1:
        return WS_TENSOR
    if sb.numel() == n and (sb.dim() == 1 or tuple(sb.shape) == (n, 1)):
        return WS_CHANNEL
    if sb.dim() == 2 and tuple(sb.shape) == (n, kblocks):
        return WS_GROUP
    raise ValueError(f"FP8 weight scales of shape {tuple(sb.shape)} match neither ({-(-n // 128)}, {kblocks}) blocks, ({n}, {kblocks}) channel"
                     f" blocks, ({n}) channels nor a tensor")


def _scales(sb):
    return sb if sb.dtype == torch.float32 and sb.is_contiguous() else sb.float().contiguous()


# Tile configurations (variant, stream-K split) timed per shape bucket, kernel path and GPU; the first is the untuned default.
_CANDIDATES = ((2, 1), (2, 0), (0, 1), (0, 0), (1, 0), (1, 1))
_TUNING_VERSION = 1
_TUNING_PATH = Path.home() / ".triton" / "autotune" / "wgpkernels_fp8.json"
_tuned = None


@functools.cache
def _device_name(index):
    return torch.cuda.get_device_name(index)


def _rows_bucket(m):
    # Tile and wave counts depend on the exact row count up to a few waves; beyond, the choice is stable.
    return -(-m // 64) * 64 if m <= 4096 else min(1 << (m - 1).bit_length(), 16384)


def _cache():
    global _tuned
    if _tuned is None:
        try:
            payload = json.loads(_TUNING_PATH.read_text(encoding="utf-8"))
            _tuned = payload["configs"] if payload.get("version") == _TUNING_VERSION else {}
        except (OSError, ValueError, KeyError):
            _tuned = {}
    return _tuned


def _save_cache():
    try:
        _TUNING_PATH.parent.mkdir(parents=True, exist_ok=True)
        _TUNING_PATH.write_text(json.dumps({"version": _TUNING_VERSION, "configs": _tuned}, indent=1), encoding="utf-8")
    except OSError as error:
        print(f"[FP8] Could not save the FP8 GEMM tuning cache {_TUNING_PATH}: {error}")


def _best_of(launches, device):
    """Best time in ms of each launch(), three rounds in alternating order (a burst of other GPU work must not decide). Before every
    timed launch, the library reads twice the L2 size from its own cudaMalloc buffer (outside PyTorch's allocator, so that tuning in the
    eager warmup of a CUDA graph leaves nothing in the graph's pool): in a model each layer streams its weights from memory. Without
    memory for that buffer, launches are timed back to back."""
    names = list(launches)
    for name in names:
        launches[name]()
    flush_bytes, stream = 2 * torch.cuda.get_device_properties(device).L2_cache_size, _stream(device)
    cold = _lib.wf8_l2_flush(device.index, flush_bytes, stream) == 0
    timings = {name: float("inf") for name in names}
    try:
        for round_index in range(3):
            for name in (names if round_index % 2 == 0 else names[::-1]):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                if cold:
                    _check(_lib.wf8_l2_flush(device.index, flush_bytes, stream))
                else:
                    torch.cuda._sleep(1_000_000)  # queue the launches so that host overhead does not reach the measurement
                start.record()
                for _ in range(1 if cold else 3):
                    launches[name]()
                end.record()
                end.synchronize()
                timings[name] = min(timings[name], start.elapsed_time(end) / (1 if cold else 3))
    finally:
        _lib.wf8_l2_flush_release()
    return timings


def _tuned_config(a, sa, b, sb, y):
    """((variant, split), whether it is a tuned choice) of this shape bucket: cached, else timed now on this call's operands (y: an
    output the timing may overwrite), or the untuned default during CUDA graph capture or compilation."""
    (m, k), n = a.shape, b.shape[0]
    key = f"block|{_rows_bucket(m)}x{n}x{k}|{kernel_path()}|{_device_name(b.device.index)}"
    config = _cache().get(key)
    if config is not None:
        return tuple(config), True
    if torch.cuda.is_current_stream_capturing() or torch.compiler.is_compiling():
        return (_lib.wf8_gemm_select(m, n, k, _sm_count(b.device)), 1), False
    started = time.time()
    label = f"FP8 GEMM {m}x{n}x{k} ({kernel_path()})"
    print(f"[WGPKernels] Autotuning {label}...", flush=True)
    candidates = _CANDIDATES if torch.version.hip is None else ((1, 0),)  # AMD: one tile, no stream-K
    timings = _best_of({config: (lambda config: lambda: gemm(a, sa, b, sb, None, y.dtype, config[1], config[0], y))(config)
                        for config in candidates}, b.device)
    best_ms, best = min((ms, config) for config, ms in timings.items())
    _tuned[key] = list(best)
    _save_cache()
    print(f"[WGPKernels] Selected {label}: tiles {best[0]}, split {best[1]} ({best_ms * 1000:.1f} us; tuned in {time.time() - started:.2f} s).", flush=True)
    return best, True


# (M, N, K, device, library) -> (variant, split, slot bytes, counter bytes) of tuned configurations: the hot path builds no key strings and
# queries no workspace sizes.
_plans = {}


def _plan(a, sa, b, sb, y, scratch_output):
    (m, k), n = a.shape, b.shape[0]
    shape = (m, n, k, b.device.index, id(_lib))
    plan = _plans.get(shape)
    if plan is not None:
        return plan
    # the timing writes its outputs: never into an output that is also the residual
    (variant, split), tuned = _tuned_config(a, sa, b, sb, torch.empty_like(y) if scratch_output else y)
    _lib.wf8_gemm_workspace(m, n, k, variant, _sm_count(b.device), split, _sizes)
    plan = (variant, split, _sizes[0], _sizes[1])
    if tuned:
        _plans[shape] = plan
    return plan


def gemm(a, sa, b, sb, bias=None, out_dtype=torch.bfloat16, split=None, variant=None, out=None, residual=None, residual_scale=None):
    """Y = sum over 128-element blocks of (A * sa) @ (B * sb).T + bias for E4M3 A (M, K) with scales sa (M, ceil(K / 128)) (from
    quantize_blocks()) and E4M3 B (N, K) with scales sb per 128 x 128 block (ceil(N / 128), ceil(K / 128)), per channel and block of K
    (N, ceil(K / 128)), per channel (N) or per tensor (1).
    K a multiple of 16. bias (N), residual (M, N) and residual_scale (N) in out_dtype (FP16 or BF16): Y = residual + residual_scale *
    (A @ B.T + bias); out may be the residual. variant (0: 128x128 tiles, 1: 64x128, 2: 128x128 with 128-element k-tiles) and split
    (stream-K) are tuned when omitted."""
    m, k = a.shape
    n = b.shape[0]
    if a.dtype not in _FP8 or b.dtype not in _FP8 or b.shape[1] != k or sa.shape != (m, -(-k // 128)):
        raise ValueError("FP8 GEMM operands: E4M3 A (M, K), FP32 scales (M, ceil(K / 128)), E4M3 B (N, K)")
    wmode = _scale_mode(sb, n, k)
    sb = _scales(sb)
    sms = _sm_count(a.device)
    y = torch.empty((m, n), dtype=out_dtype, device=a.device) if out is None else out
    if variant is None and split is None:
        variant, split, slot_bytes, counter_bytes = _plan(a, sa, b, sb, y, residual is not None)
    else:
        variant = 2 if variant is None else variant
        split = 1 if split is None else split
        _lib.wf8_gemm_workspace(m, n, k, variant, sms, split, _sizes)
        slot_bytes, counter_bytes = _sizes[0], _sizes[1]
    slots = torch.empty(slot_bytes, dtype=torch.uint8, device=a.device) if slot_bytes else None
    counters = _zeroed_counters(a.device, counter_bytes).data_ptr() if counter_bytes else None
    _check(_lib.wf8_gemm(a.data_ptr(), sa.data_ptr(), b.data_ptr(), sb.data_ptr(), wmode, _ptr(bias), _ptr(residual), _ptr(residual_scale),
                         y.data_ptr(), m, n, k, _DTYPES[out_dtype], variant, sms, split, _ptr(slots), counters, _stream(a.device)))
    return y


def linear(x, weight, scale, bias=None, tile=False, residual=None, residual_scale=None, out=None):
    """Y = X @ (E4M3 weight * scale).T + bias for 2-D FP16/BF16 X (rows 16-byte aligned, unit column stride; K a multiple of 16), the
    activation quantized by quantize_blocks(x, tile) in row chunks within SCRATCH_BYTES. scale: FP32 per 128 x 128 block (ceil(N / 128),
    ceil(K / 128)), per channel and block of K (N, ceil(K / 128)), per channel (N) or per tensor. Residual: Y = residual + residual_scale *
    (...) in X's dtype (contiguous; out may be the residual)."""
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device) if out is None else out
    rows = max(128, SCRATCH_BYTES // (k + 4 * -(-k // 128)) // 128 * 128)  # multiples of 128: whole activation tiles
    for start in range(0, m, rows):
        stop = min(start + rows, m)
        q, sa = quantize_blocks(x[start:stop], tile)
        gemm(q, sa, weight, scale, bias, x.dtype, out=out[start:stop], residual=None if residual is None else residual[start:stop],
             residual_scale=residual_scale)
        del q, sa
    return out


def dequantize(q, scale, dtype=torch.float32):
    """Values of E4M3 q (M, K) with scales per row and block (M, ceil(K / 128)), per 128 x 128 block, per row (M) or per tensor, computed
    in dtype (FP64: exact)."""
    m, k = q.shape
    v = (q.view(torch.float8_e4m3fn) if q.dtype != torch.float8_e4m3fn else q).to(dtype)
    scale = scale.to(dtype)
    if scale.numel() == 1:
        return v * scale.reshape(())
    if scale.dim() == 2 and scale.shape[0] == m and scale.shape[1] == -(-k // 128):
        return v * scale.repeat_interleave(128, dim=1)[:, :k]
    if scale.dim() == 2 and scale.shape == (-(-m // 128), -(-k // 128)):
        return v * scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)[:m, :k]
    return v * scale.reshape(m, 1)


def reference(a, sa, b, sb, bias=None, residual=None, residual_scale=None):
    """FP64 reference of gemm() with its block semantics (the exact sum of each block's products, scaled by sa * sb)."""
    m, k = a.shape
    n = b.shape[0]
    kb = -(-k // 128)
    av = a.view(torch.float8_e4m3fn).double() if a.dtype != torch.float8_e4m3fn else a.double()
    bv = b.view(torch.float8_e4m3fn).double() if b.dtype != torch.float8_e4m3fn else b.double()
    pad = kb * 128 - k
    av = torch.nn.functional.pad(av, (0, pad)).view(m, kb, 128).transpose(0, 1)
    bv = torch.nn.functional.pad(bv, (0, pad)).view(n, kb, 128).transpose(0, 1)
    blocks = torch.bmm(av, bv.transpose(1, 2))  # (kb, M, N)
    scales = sa.double().T[:, :, None]  # (kb, M, 1)
    mode = _scale_mode(sb, n, k)
    if mode == WS_BLOCK:
        scales = scales * sb.double().T.repeat_interleave(128, dim=1)[:, None, :n]
    elif mode == WS_GROUP:
        scales = scales * sb.double().T[:, None, :]
    out = (blocks * scales).sum(0)
    if mode == WS_CHANNEL:
        out = out * sb.double().reshape(1, n)
    elif mode == WS_TENSOR:
        out = out * sb.double().reshape(())
    if bias is not None:
        out = out + bias.double()
    if residual is not None:
        out = residual.double() + (1 if residual_scale is None else residual_scale.double()) * out
    return out


def selftest(device=None):
    """Runs the quantizer and every tile configuration once on small inputs (no tuning): None when the results match an FP64 reference
    of the same block semantics, else a reason."""
    device = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
    generator = torch.Generator(device=device).manual_seed(0)
    for dtype in (torch.bfloat16, torch.float16):
        x = torch.randn(200, 384, dtype=dtype, device=device, generator=generator)
        x[:, 5] *= 30  # an outlier channel
        q, sa = quantize_blocks(x)
        reference_q = (x.float().view(200, 3, 128) * (1.0 / (x.float().abs().view(200, 3, 128).amax(-1, keepdim=True).clamp(1e-4) / 448.0))).to(torch.float8_e4m3fn)
        if not torch.equal(q.view(torch.uint8), reference_q.view(200, 384).view(torch.uint8)):
            return f"quantize_blocks mismatch ({dtype})"
        w = torch.randn(160, 384, device=device, generator=generator).to(dtype)
        wq, ws = quantize_weight(w)
        bias = torch.randn(160, dtype=dtype, device=device, generator=generator)
        residual = torch.randn(200, 160, dtype=dtype, device=device, generator=generator)
        residual_scale = torch.rand(160, dtype=dtype, device=device, generator=generator)
        channel = torch.rand(160, device=device, generator=generator) / 448
        groups = torch.rand(160, 3, device=device, generator=generator) / 448
        for variant in (0, 1, 2):
            for split in (0, 1):
                for sb, extra in ((ws, False), (channel, True), (channel[:1], False), (groups, True)):
                    r, rs = (residual.clone(), residual_scale) if extra else (None, None)
                    ref = reference(q, sa, wq, sb, bias, r, rs)
                    y = gemm(q, sa, wq, sb, bias, dtype, split, variant, out=r, residual=r, residual_scale=rs)
                    error = ((y.double() - ref).norm() / ref.norm()).item()
                    if not error < 5e-3:
                        return f"FP8 GEMM mismatch ({dtype}, tiles {variant}, split {split}, scales {tuple(sb.shape)}): relative error {error:.3g}"
    torch.cuda.synchronize(device)
    return None
