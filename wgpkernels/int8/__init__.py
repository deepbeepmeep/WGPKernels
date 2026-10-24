# Copyright (c) 2026 DeepBeepMeep - Apache License 2.0 (see LICENSE.txt)
"""INT8 weight linears (Quanto INT8 and ConvRot checkpoints): Y = X @ (W * per-channel FP32 scales).T + bias, W the checkpoint's INT8
(N, K) matrix, never dequantized or copied. Kernels come from an ahead-of-time built library (python -m wgpkernels.build).

Paths, by precision:
- full precision (as precise as unquantized activations):
  'lines' (1-4 tokens): CUDA-core GEMV, one warp per output channel; 'small' (up to 32 tokens): tensor cores with INT8 weights
  converted exactly to BF16/FP16; 'exact': INT8 x INT8 tensor cores on an exact INT8 split of every activation
  (x ~= scale * (hi + lo / 254)), twice the tensor work of one pass;
- 'block': one INT8 pass with one FP32 scale per token row and 128-element block of K (ConvRot rotation first): an outlier only
  coarsens its own block;
- 'row': one INT8 pass with one scale per row (classic W8A8 precision), only on request.
linear() takes the fastest full-precision path while it costs at most EXACT_TOLERANCE more than 'block', else 'block'. Whether that
holds depends on the GPU's compute to memory bandwidth ratio, so the choice is timed per GPU, shape and token-count bucket on the
first call and cached (fixed thresholds measured on an RTX 5090 apply during CUDA graph capture or compilation).
precise=True keeps the fastest full-precision path at every token count (layers whose activations need it).
Activations are quantized in row chunks bounded by SCRATCH_BYTES, so no full-size quantized copy is allocated.

Group-quantized weights (one FP32 scale per output channel and 128-element block of K, scale (N, ceil(K / 128))) run the 'block' and
'exact' paths: each block's exact INT32 sums are promoted with their weight scales into FP32 accumulators.

Fusions (every path): an input operation applied in FP32 where the activation is quantized or rotated, never stored ('rms_norm': RMSNorm
with an optional weight; 'swiglu': silu(gate) * value of rows [gate | value]; 'gelu': GELU, tanh approximation), and a residual epilogue
Y = R + scale * (X @ W.T + bias)
computed in FP32 and rounded once, written in place of R when out is R. prepare() quantizes an activation once for several weights or
row ranges of a weight (linear_prepared()).
"""
import ctypes
import functools
import json
import os
import time
from contextvars import ContextVar
from pathlib import Path

import torch

SCRATCH_BYTES = 16 * 1024 * 1024
# The block quantizer and group rotation put token rows on CUDA's grid Y axis.
MAX_GRID_Y = 65535
EXACT_TOLERANCE = 0.10
# Fallback thresholds (RTX 5090) when a choice cannot be timed; candidate limits of the few-token paths.
EXACT_ROWS, LINE_ROWS, SMALL_ROWS = 64, 2, 16
LINE_MAX_ROWS, SMALL_MAX_ROWS, DECIDE_MAX_ROWS = 4, 32, 4096
ACTS = {None: 0, "rms_norm": 1, "swiglu": 2, "gelu": 3}
_COUNTERS = 1024  # split tiles never exceed the SM count (GEMM) or twice of it (few-token kernel)
HIP_ARCHS = ("gfx1200", "gfx1201")  # the AMD targets of the HIP library (RDNA4)
_lib = None
_weight_groups = False  # the library takes per-(n, k-block) weight scales (wq8_features bit 0)
_DTYPES = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}
_sms = {}
_counters = {}
_sizes = (ctypes.c_int64 * 2)()


def default_library_path():
    return Path(__file__).with_name("wgp_int8.dll" if os.name == "nt" else "libwgp_int8.so")


def weight_groups_supported():
    """Whether the loaded library takes per-(n, 128-element k-block) weight scales (N, ceil(K / 128))."""
    return _weight_groups


def load_library(path=None):
    global _lib, _weight_groups
    lib = ctypes.CDLL(str(Path(path or default_library_path()).resolve()))
    if lib.wq8_abi() != 11:
        raise RuntimeError("INT8 GEMM library ABI mismatch; rebuild it")
    features = getattr(lib, "wq8_features", None)
    _weight_groups = bool(features and features() & 1)
    p, i, i64, f = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_float
    lib.wq8_quantize.argtypes = [p, p, p, p, i, i, i64, i, i, i, p, f, p]
    lib.wq8_quantize_blocks.argtypes = [p, p, p, i, i, i64, i, i, i, p, f, p, p]
    lib.wq8_transform_rows.argtypes = [p, p, i, i, i64, i, i, i, p, f, p]
    lib.wq8_l2_flush.argtypes = [i, i64, p]
    lib.wq8_l2_flush_release.restype = None
    lib.wq8_gemm.argtypes = [p, p, p, p, p, p, p, p, p, i, i, i, i, i, i, i, p, p, p]
    lib.wq8_gemm_workspace.argtypes = [i, i, i, i, i, i, p]
    lib.wq8_gemm_workspace.restype = None
    lib.wq8_gemm_select.argtypes = [i, i, i, i, i]
    lib.wq8_small.argtypes = [p, i64, p, p, p, p, p, p, i, i, i, i, i, p, p, p]
    lib.wq8_small_workspace.argtypes = [i, i, i, i, p]
    lib.wq8_small_workspace.restype = None
    lib.wq8_gemv_lines.argtypes = [p, i64, p, p, p, p, p, p, i, i, i, i, p]
    lib.wq8_rotate_groups.argtypes = [p, p, i, i, i64, i, i, p]
    for index in range(torch.cuda.device_count()):
        # ROCm: only the GPUs the library is built for (PyTorch has no kernels for others, e.g. a Ryzen integrated GPU beside the RDNA4 one)
        if torch.version.hip is None or torch.cuda.get_device_properties(index).gcnArchName.split(":")[0] in HIP_ARCHS:
            _counters.setdefault(index, torch.zeros(_COUNTERS, dtype=torch.int32, device=torch.device("cuda", index)))
    _lib = lib
    return lib


def _check(code):
    if code:
        raise RuntimeError(f"INT8 GEMM launch failed (CUDA error {code})")


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
        raise RuntimeError("INT8 GEMM split-K counters exceeded")
    return _counters[device.index]


def _width(x, act):
    """K of the linear fed by x: SwiGLU rows hold the gate and value halves."""
    return x.shape[1] // 2 if act == "swiglu" else x.shape[1]


def quantize_blocks(x, group=0, act=None, act_weight=None, eps=0.0):
    """INT8 of op(x) (optionally ConvRot-rotated) with one FP32 scale per row and 128-element block: (q, scales (M, ceil(K / 128)))."""
    m, k = x.shape[0], _width(x, act)
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    scale = torch.empty((m, (k + 127) // 128), dtype=torch.float32, device=x.device)
    inv_rms = torch.empty(m, dtype=torch.float32, device=x.device) if act == "rms_norm" and k > 16384 else None
    x_ptr, q_ptr, scale_ptr, rms_ptr = x.data_ptr(), q.data_ptr(), scale.data_ptr(), _ptr(inv_rms)
    ldx, itemsize, scale_columns = x.stride(0), x.element_size(), scale.shape[1]
    stream = _stream(x.device)
    for start in range(0, m, MAX_GRID_Y):
        rows = min(MAX_GRID_Y, m - start)
        _check(_lib.wq8_quantize_blocks(x_ptr + start * ldx * itemsize, q_ptr + start * k, scale_ptr + start * scale_columns * 4, rows, k, ldx,
                                        _DTYPES[x.dtype], group, ACTS[act], _ptr(act_weight), eps, None if rms_ptr is None else rms_ptr + start * 4, stream))
    return q, scale


def quantize(x, group=0, hilo=False, act=None, act_weight=None, eps=0.0):
    """Row-wise symmetric INT8 of op(x) (optionally ConvRot-rotated): returns (hi, lo or None, FP32 row scales)."""
    m, k = x.shape[0], _width(x, act)
    hi = torch.empty((m, k), dtype=torch.int8, device=x.device)
    lo = torch.empty_like(hi) if hilo else None
    scale = torch.empty(m, dtype=torch.float32, device=x.device)
    _check(_lib.wq8_quantize(x.data_ptr(), hi.data_ptr(), _ptr(lo), scale.data_ptr(), m, k, x.stride(0), _DTYPES[x.dtype], group, ACTS[act],
                             _ptr(act_weight), eps, _stream(x.device)))
    return hi, lo, scale


def transform_rows(x, group=0, act=None, act_weight=None, eps=0.0):
    """op(x), ConvRot-rotated, rounded once to x's dtype: the unquantized activation of the few-token paths."""
    m, k = x.shape[0], _width(x, act)
    out = torch.empty((m, k), dtype=x.dtype, device=x.device)
    _check(_lib.wq8_transform_rows(x.data_ptr(), out.data_ptr(), m, k, x.stride(0), _DTYPES[x.dtype], group, ACTS[act], _ptr(act_weight), eps,
                                   _stream(x.device)))
    return out


# Tile configurations (variant, stream-K split) timed per shape bucket and GPU; the first is the untuned default.
_CANDIDATES = {'row': ((0, 1), (0, 0), (2, 0), (2, 1), (6, 0), (6, 1)), 'exact': ((1, 0), (1, 1), (7, 0), (7, 1)), 'block': ((10, 1), (10, 0), (11, 0), (11, 1)),
               # per-(n, k-block) weight scales
               'exact+g': ((14, 1), (14, 0)), 'block+g': ((12, 1), (12, 0), (13, 0), (13, 1))}
_MODES = {0: 'row', 2: 'row', 6: 'row', 1: 'exact', 7: 'exact', 10: 'block', 11: 'block', 12: 'block+g', 13: 'block+g', 14: 'exact+g'}
_TUNING_VERSION = 12
_TUNING_PATH = Path.home() / ".triton" / "autotune" / "wgpkernels_int8.json"
_tuned = None
_path_tuning = ContextVar("wgpkernels_int8_path_tuning", default=False)


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
        print(f"[INT8] Could not save the INT8 GEMM tuning cache {_TUNING_PATH}: {error}")


def _best_of(launches, device):
    """Best time in ms of each launch(), three rounds in alternating order (a burst of other GPU work must not decide). Before every
    timed launch, the library reads twice the L2 size from its own cudaMalloc buffer (outside PyTorch's allocator, so that tuning in
    the eager warmup of a CUDA graph leaves nothing in the graph's pool): in a model each layer streams its weights from memory, which
    favors memory-bound paths more than repeated warm launches would. Without memory for that buffer, launches are timed back to back."""
    names = list(launches)
    for name in names:
        launches[name]()
    flush_bytes, stream = 2 * torch.cuda.get_device_properties(device).L2_cache_size, _stream(device)
    cold = _lib.wq8_l2_flush(device.index, flush_bytes, stream) == 0
    timings = {name: float("inf") for name in names}
    try:
        for round_index in range(3):
            for name in (names if round_index % 2 == 0 else names[::-1]):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                if cold:
                    _check(_lib.wq8_l2_flush(device.index, flush_bytes, stream))
                else:
                    torch.cuda._sleep(1_000_000)  # queue the launches so that host overhead does not reach the measurement
                start.record()
                for _ in range(1 if cold else 3):
                    launches[name]()
                end.record()
                end.synchronize()
                timings[name] = min(timings[name], start.elapsed_time(end) / (1 if cold else 3))
    finally:
        _lib.wq8_l2_flush_release()
    return timings


def _tuned_config(a, a_lo, sa, b, sb, bias, y, mode):
    """((variant, split), whether it is a tuned choice) of this shape bucket: cached, else timed now on this call's operands (y: an
    output the timing may overwrite), or the untuned default during CUDA graph capture or compilation."""
    (m, k), n = a.shape, b.shape[0]
    key = f"{mode}|{_rows_bucket(m)}x{n}x{k}|{_device_name(b.device.index)}"
    config = _cache().get(key)
    if config is not None:
        return tuple(config), True
    if torch.cuda.is_current_stream_capturing() or torch.compiler.is_compiling():
        if mode in ('block', 'block+g', 'exact+g'):
            return _CANDIDATES[mode][0], False
        return (_lib.wq8_gemm_select(m, n, k, mode == 'exact', _sm_count(b.device)), 1), False
    started = time.time()
    log = not _path_tuning.get()
    label = f"INT8 GEMM {mode} {m}x{n}x{k}"
    if log:
        print(f"[WGPKernels] Autotuning {label}...", flush=True)
    timings = _best_of({config: (lambda config: lambda: gemm(a, a_lo, sa, b, sb, bias, y.dtype, config[1], config[0], y))(config)
                        for config in _CANDIDATES[mode]}, b.device)
    best_ms, best = min((ms, config) for config, ms in timings.items())
    _tuned[key] = list(best)
    _save_cache()
    if log:
        print(f"[WGPKernels] Selected {label}: tiles {best[0]}, split {best[1]} ({best_ms * 1000:.1f} us; tuned in {time.time() - started:.2f} s).", flush=True)
    return best, True


# (mode, M, N, K, device) -> (variant, split, slot bytes, counter bytes) of tuned configurations: the hot path builds no key strings and
# queries no workspace sizes.
_plans = {}


def _plan(a, a_lo, sa, b, sb, bias, y, mode, scratch_output):
    (m, k), n = a.shape, b.shape[0]
    shape = (mode, m, n, k, b.device.index)
    plan = _plans.get(shape)
    if plan is not None:
        return plan
    # the timing writes its outputs: never into an output that is also the residual
    (variant, split), tuned = _tuned_config(a, a_lo, sa, b, sb, bias, torch.empty_like(y) if scratch_output else y, mode)
    _lib.wq8_gemm_workspace(m, n, k, variant, _sm_count(b.device), split, _sizes)
    plan = (variant, split, _sizes[0], _sizes[1])
    if tuned:
        _plans[shape] = plan
    return plan


def _weight_scale_groups(sb):
    """Whether sb holds one scale per output channel and 128-element block of K (N, ceil(K / 128)), rather than per channel (N)."""
    if sb.dim() != 2:
        return False
    if not _weight_groups:
        raise RuntimeError("This INT8 library predates per-(n, k-block) weight scales; rebuild it")
    return True


def gemm(a, a_lo, sa, b, sb, bias, out_dtype, split=None, variant=None, out=None, mode=None, residual=None, residual_scale=None):
    """mode: 'row' (sa per row), 'exact' (a_lo residual) or 'block' (sa per row and 128-element block); inferred when omitted. sb: per
    output channel (N), or per channel and 128-element block of K (N, ceil(K / 128)) with the 'exact' and 'block' modes (variants 12-14).
    residual (M, N) and residual_scale (N) in out_dtype: Y = residual + residual_scale * (A @ B.T + bias); out may be the residual."""
    m, k = a.shape
    n = b.shape[0]
    sms = _sm_count(a.device)
    y = torch.empty((m, n), dtype=out_dtype, device=a.device) if out is None else out
    if variant is None and split is None:
        mode = mode or ('exact' if a_lo is not None else 'block' if sa.dim() == 2 else 'row')
        if _weight_scale_groups(sb):
            if mode == 'row':
                raise ValueError("Weights with per-(n, k-block) scales run the 'block' or 'exact' path")
            mode += '+g'
        variant, split, slot_bytes, counter_bytes = _plan(a, a_lo, sa, b, sb, bias, y, mode, residual is not None)
    else:
        split = 1 if split is None else split
        _lib.wq8_gemm_workspace(m, n, k, variant, sms, split, _sizes)
        slot_bytes, counter_bytes = _sizes[0], _sizes[1]
    slots = torch.empty(slot_bytes, dtype=torch.uint8, device=a.device) if slot_bytes else None
    counters = _zeroed_counters(a.device, counter_bytes).data_ptr() if counter_bytes else None
    _check(_lib.wq8_gemm(a.data_ptr(), _ptr(a_lo), sa.data_ptr(), b.data_ptr(), sb.data_ptr(), _ptr(bias), _ptr(residual), _ptr(residual_scale),
                         y.data_ptr(), m, n, k, _DTYPES[out_dtype], variant, sms, split, _ptr(slots), counters, _stream(a.device)))
    return y


def small(x, weight, scale, bias=None, residual=None, residual_scale=None, out=None):
    """Weight-only INT8 for at most 32 tokens: unquantized activations, FP32 accumulation. K multiple of 64."""
    m, k = x.shape
    n = weight.shape[0]
    sms = _sm_count(x.device)
    y = torch.empty((m, n), dtype=x.dtype, device=x.device) if out is None else out
    _lib.wq8_small_workspace(m, n, k, sms, _sizes)
    slots = torch.empty(_sizes[0] // 4, dtype=torch.float32, device=x.device) if _sizes[0] else None
    counters = _zeroed_counters(x.device, _sizes[1]).data_ptr() if _sizes[1] else None
    _check(_lib.wq8_small(x.data_ptr(), x.stride(0), weight.data_ptr(), scale.data_ptr(), _ptr(bias), _ptr(residual), _ptr(residual_scale),
                          y.data_ptr(), m, n, k, _DTYPES[x.dtype], sms, _ptr(slots), counters, _stream(x.device)))
    return y


def gemv_lines(x, weight, scale, bias=None, residual=None, residual_scale=None, out=None):
    """One to four tokens, unquantized activations: one warp per output channel, 128-byte row lines per load (K multiple of 4)."""
    m, k = x.shape
    y = torch.empty((m, weight.shape[0]), dtype=x.dtype, device=x.device) if out is None else out
    _check(_lib.wq8_gemv_lines(x.data_ptr(), x.stride(0), weight.data_ptr(), scale.data_ptr(), _ptr(bias), _ptr(residual), _ptr(residual_scale),
                               y.data_ptr(), m, weight.shape[0], k, _DTYPES[x.dtype], _stream(x.device)))
    return y


def rotate_groups(x, group):
    """ConvRot rotation (groups of 16, 64 or 256), rounded once to the input dtype: the few-token paths keep activations unquantized."""
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    x_ptr, out_ptr, itemsize, ldx = x.data_ptr(), out.data_ptr(), x.element_size(), x.stride(0)
    stream = _stream(x.device)
    for start in range(0, x.shape[0], MAX_GRID_Y):
        rows = min(MAX_GRID_Y, x.shape[0] - start)
        _check(_lib.wq8_rotate_groups(x_ptr + start * ldx * itemsize, out_ptr + start * x.shape[1] * itemsize, rows, x.shape[1], ldx, _DTYPES[x.dtype], group, stream))
    return out


class Prepared:
    """An activation prepared for one path: the unquantized (transformed, rotated) rows of 'lines' / 'small', or the INT8 rows and scales
    of 'exact' / 'block' / 'row'. Rows can be sliced; the same preparation serves every weight with the same K and group."""
    __slots__ = ("path", "x", "lo", "scale", "dtype")

    def __init__(self, path, x, lo, scale, dtype):
        self.path, self.x, self.lo, self.scale, self.dtype = path, x, lo, scale, dtype

    def rows(self, start, stop):
        part = lambda t: None if t is None else t[start:stop]
        return Prepared(self.path, self.x[start:stop], part(self.lo), part(self.scale), self.dtype)


def prepare(x, group=0, path="block", act=None, act_weight=None, eps=0.0):
    """op(x) prepared once for `path` (choose_path()); x: 2-D BF16/FP16, rows 16-byte aligned. act_weight in x's dtype."""
    if path in ("lines", "small"):
        prepared = transform_rows(x, group, act, act_weight, eps) if act else rotate_groups(x, group) if group else x
        return Prepared(path, prepared, None, None, x.dtype)
    if path == "block":
        q, scale = quantize_blocks(x, group, act, act_weight, eps)
        return Prepared(path, q, None, scale, x.dtype)
    hi, lo, scale = quantize(x, group, path == "exact", act, act_weight, eps)
    return Prepared(path, hi, lo, scale, x.dtype)


def linear_prepared(prepared, weight, scale, bias=None, residual=None, residual_scale=None, out=None):
    """Y = prepared @ (weight * scale).T + bias (or residual + residual_scale * (...)), weight an INT8 (N, K) matrix or row range."""
    if prepared.path in ("lines", "small") and scale.dim() == 2:
        raise ValueError("Weights with per-(n, k-block) scales run the 'block' or 'exact' path")
    if prepared.path == "lines":
        return gemv_lines(prepared.x, weight, scale, bias, residual, residual_scale, out)
    if prepared.path == "small":
        return small(prepared.x, weight, scale, bias, residual, residual_scale, out)
    return gemm(prepared.x, prepared.lo, prepared.scale, weight, scale, bias, prepared.dtype, out=out, mode=prepared.path, residual=residual,
                residual_scale=residual_scale)


def _gemm_path(x, weight, scale, bias, group, mode, act=None, act_weight=None, eps=0.0, residual=None, residual_scale=None, out=None):
    """'exact', 'block' or 'row' over row chunks whose quantized activations stay within SCRATCH_BYTES."""
    m, k = x.shape[0], weight.shape[1]
    out = torch.empty((m, weight.shape[0]), dtype=x.dtype, device=x.device) if out is None else out
    row_bytes = (2 * k if mode == 'exact' else k) + 4 * (-(-k // 128) if mode == 'block' else 1)
    rows = max(128, SCRATCH_BYTES // row_bytes // 128 * 128)
    for start in range(0, m, rows):
        stop = min(start + rows, m)
        prepared = prepare(x[start:stop], group, mode, act, act_weight, eps)
        linear_prepared(prepared, weight, scale, bias, None if residual is None else residual[start:stop], residual_scale, out[start:stop])
        del prepared
    return out


def _run(path, x, weight, scale, bias, group, act=None, act_weight=None, eps=0.0, residual=None, residual_scale=None, out=None):
    if path in ('lines', 'small'):
        return linear_prepared(prepare(x, group, path, act, act_weight, eps), weight, scale, bias, residual, residual_scale, out)
    return _gemm_path(x, weight, scale, bias, group, path, act, act_weight, eps, residual, residual_scale, out)


def _precise_paths(m, k, groups=False):
    """Full-precision candidates; weights with per-(n, k-block) scales have only 'exact' (the few-token paths take channel scales)."""
    return [path for path, ok in (('lines', m <= LINE_MAX_ROWS and k % 4 == 0 and not groups), ('small', m <= SMALL_MAX_ROWS and k % 64 == 0 and not groups),
                                  ('exact', True)) if ok]


def _static_path(m, k, precise, groups=False):
    if m <= LINE_ROWS and k % 4 == 0 and not groups:
        return 'lines'
    if m <= SMALL_ROWS and k % 64 == 0 and not groups:
        return 'small'
    return 'exact' if precise or m <= EXACT_ROWS else 'block'


def _path_bucket(m):
    return m if m <= 16 else -(-m // 8) * 8 if m <= 128 else _rows_bucket(m)


_paths = {}  # (M, N, K, group, act, precise, device) -> chosen path, as _plans


def choose_path(x, weight, scale, bias=None, group=0, precise=False, act=None, act_weight=None, eps=0.0):
    """Fastest full-precision path while it costs at most EXACT_TOLERANCE more than 'block' (always with precise=True), timed on
    this call's input per GPU, shape, input operation and token-count bucket."""
    (m, k), n = (x.shape[0], weight.shape[1]), weight.shape[0]
    if m > DECIDE_MAX_ROWS and not precise:
        return 'block'  # two INT8 passes never come close to one at this size
    groups = _weight_scale_groups(scale)
    shape = (m, n, k, group, act, precise, x.device.index, groups)
    path = _paths.get(shape)
    if path is not None:
        return path
    key = (f"path{'+precise' if precise else ''}|{_path_bucket(m)}x{n}x{k}|g{group}{f'|{act}' if act else ''}{'|wgroups' if groups else ''}"
           f"|{_device_name(x.device.index)}")
    path = _cache().get(key)
    if path is not None:
        _paths[shape] = path
        return path
    if torch.cuda.is_current_stream_capturing() or torch.compiler.is_compiling():
        return _static_path(m, k, precise, groups)
    candidates = _precise_paths(m, k, groups) + ([] if precise else ['block'])
    label = f"INT8 path {m}x{n}x{k}{f' ConvRot {group}' if group else ''}{f' {act}' if act else ''}{' (precise)' if precise else ''}"
    started = time.time()
    print(f"[WGPKernels] Autotuning {label}...", flush=True)
    token = _path_tuning.set(True)
    try:
        timings = _best_of({path: (lambda path: lambda: _run(path, x, weight, scale, bias, group, act, act_weight, eps))(path) for path in candidates},
                           x.device)
    finally:
        _path_tuning.reset(token)
    full_ms, path = min((timings[p], p) for p in candidates if p != 'block')
    if not precise and full_ms > (1 + EXACT_TOLERANCE) * timings['block']:
        path = 'block'
    _tuned[key] = _paths[shape] = path
    _save_cache()
    print(f"[WGPKernels] Selected {label}: {path} ({timings[path] * 1000:.1f} us; tuned in {time.time() - started:.2f} s).", flush=True)
    return path


def linear(x, weight, scale, bias=None, group=0, precise=False, mode=None, act=None, act_weight=None, eps=0.0, residual=None, residual_scale=None,
           out=None):
    """Y = op(X) @ (INT8 weight * FP32 channel scales).T + bias for 2-D BF16/FP16 X (rows 16-byte aligned, unit column stride), or
    residual + residual_scale * (...) (contiguous, X's dtype; out may be the residual). act: None, 'rms_norm' (act_weight: K values in X's
    dtype or None; eps), 'swiglu' (X rows: gate | value) or 'gelu' (tanh approximation). mode forces a path ('lines', 'small', 'exact',
    'block', 'row'); by default
    the path is chosen as described above."""
    path = mode or choose_path(x, weight, scale, bias, group, precise, act, act_weight, eps)
    return _run(path, x, weight, scale, bias, group, act, act_weight, eps, residual, residual_scale, out)


def _reference(x, w, s, bias, group, act, act_weight, eps, residual, residual_scale, round_input):
    """FP32 reference of linear(); round_input rounds op(x) to x's dtype as the few-token paths do."""
    v = x.float()
    if act == "swiglu":
        gate, value = v.chunk(2, dim=-1)
        v = torch.nn.functional.silu(gate) * value
    elif act == "rms_norm":
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps) * (1 if act_weight is None else act_weight.float())
    elif act == "gelu":
        v = torch.nn.functional.gelu(v, approximate="tanh")
    if group:
        v = _reference_rotation(v, group)
    if round_input:
        v = v.to(x.dtype).float()
    scale = s.repeat_interleave(128, dim=1)[:, :w.shape[1]] if s.dim() == 2 else s[:, None]  # per (n, k-block) or per channel
    out = v @ (w.float() * scale).T + bias.float()
    return out if residual is None else residual.float() + (1 if residual_scale is None else residual_scale.float()) * out


def selftest(device=None):
    """Runs every kernel once on small inputs with fixed configurations (no tuning): None when the results match PyTorch, else a reason."""
    device = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
    generator = torch.Generator(device=device).manual_seed(0)
    variants = {"exact": 7, "block": 11, "row": 6, "exact+g": 14, "block+g": 13}
    cases = ((1, "lines", 0, None, False), (5, "small", 0, None, False), (5, "small", 64, None, False), (40, "exact", 0, None, False),
             (40, "block", 256, None, False), (40, "row", 16, None, False), (2, "lines", 0, "swiglu", True), (5, "small", 64, "rms_norm", True),
             (40, "exact", 256, "rms_norm", False), (40, "block", 0, "swiglu", True), (40, "block", 16, "rms_norm", True), (40, "exact", 64, None, True),
             (3, "lines", 64, "gelu", False), (5, "small", 0, "gelu", True), (40, "exact", 16, "gelu", False), (40, "block", 256, "gelu", True))
    if _weight_groups:  # weights with per-(n, k-block) scales
        cases += ((40, "exact+g", 0, None, True), (40, "block+g", 64, "rms_norm", False), (40, "block+g", 0, None, True))
    for dtype in (torch.bfloat16, torch.float16):
        w = torch.randint(-127, 128, (96, 256), dtype=torch.int8, device=device, generator=generator)
        s = torch.rand(96, device=device, generator=generator) / 127
        s_groups = torch.rand(96, 2, device=device, generator=generator) / 127
        bias = torch.randn(96, dtype=dtype, device=device, generator=generator)
        norm_weight = (torch.rand(256, device=device, generator=generator) + 0.5).to(dtype)
        for m, path, group, act, with_residual in cases:
            x = torch.randn(m, 512 if act == "swiglu" else 256, dtype=dtype, device=device, generator=generator)
            act_weight = norm_weight if act == "rms_norm" else None
            residual = torch.randn(m, 96, dtype=dtype, device=device, generator=generator) * 4 if with_residual else None
            residual_scale = torch.rand(96, dtype=dtype, device=device, generator=generator) if with_residual else None
            scale = s_groups if path.endswith("+g") else s
            ref = _reference(x, w, scale, bias, group, act, act_weight, 1e-6, residual, residual_scale, path in ("lines", "small"))
            prepared = prepare(x, group, path.removesuffix("+g"), act, act_weight, 1e-6)
            if path in ("lines", "small"):
                y = linear_prepared(prepared, w, scale, bias, residual, residual_scale, residual)  # in place of the residual
            else:
                y = gemm(prepared.x, prepared.lo, prepared.scale, w, scale, bias, dtype, 0, variants[path], out=residual, residual=residual,
                         residual_scale=residual_scale)
            error = ((y.float() - ref).norm() / ref.norm()).item()
            if not error < (5e-3 if path in ("lines", "small", "exact", "exact+g") else 3e-2):
                return f"{path} kernel mismatch ({dtype}, {m} rows, group {group}, {act or 'no input op'}{', residual' if with_residual else ''}): relative error {error:.3g}"
    torch.cuda.synchronize(device)
    return None


def _reference_rotation(v, group):
    h = torch.tensor([[1., 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], device=v.device)
    rotation = h
    while rotation.shape[0] < group:
        rotation = torch.kron(rotation, h)
    return (v.reshape(v.shape[0], -1, group) @ (rotation * group ** -0.5)).reshape(v.shape)
