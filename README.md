# WGPKernels

Precise and fast GPU kernels for [WanGP](https://github.com/deepbeepmeep/Wan2GP), usable from any PyTorch program.

Version 1 provides **INT8 weight linears** for Quanto INT8 and ConvRot checkpoints: `Y = X @ (W * scales).T + bias`, where `W` is the
checkpoint's INT8 matrix (never dequantized or copied) and `scales` its per-channel FP32 scales. Unlike classic W8A8 kernels, which give
every token a single activation scale (one large activation value then erases the small ones of its row), WGPKernels keep activations
at full precision for short inputs and use one scale per block of 128 values for long inputs.

The kernels are built ahead of time into a C library loaded with `ctypes`: no Python extension, no dependency on a Python, PyTorch or
CUDA toolkit version at run time. PyTorch supplies the tensors and the current stream.

## Precision paths

| Path | Used for | Precision |
|---|---|---|
| `lines` | 1-4 tokens | full precision activations, CUDA-core GEMV, one warp per output channel |
| `small` | up to 32 tokens | full precision activations, tensor cores on INT8 weights converted exactly to BF16/FP16 |
| `exact` | any length | INT8 tensor cores on an exact split of every activation (`x = scale * (hi + lo / 254)`), twice the tensor work |
| `block` | long inputs | one INT8 pass, one FP32 scale per token and 128-element block: an outlier only coarsens its own block |
| `row` | on request only | classic W8A8 (one scale per token) |

`linear()` uses the fastest full-precision path while it costs at most 10% more than `block`, else `block`. The choice depends on the
GPU, so it is timed once per GPU, layer shape and token-count bucket, and cached in `~/.triton/autotune/wgpkernels_int8.json`.
`precise=True` keeps a full-precision path at every length, for layers whose activations need it (for example language models with
massive activations).

Fusions, available on every path and computed in FP32 with a single final rounding:
- input operation applied while the activation is quantized: `act="rms_norm"` (optional weight) or `act="swiglu"` (rows `[gate | value]`);
- residual epilogue: `Y = residual + residual_scale * (X @ W.T + bias)`, optionally written in place of the residual;
- `prepare()` quantizes an activation once for several weights or row ranges (`linear_prepared()`).

Measured on an RTX 5090 (whole linears, activation quantization included): `block` is about 1.65x faster than Triton INT8 kernels
with per-block activation scales at similar precision, and about 25% slower than per-token (W8A8) kernels on large matrix products,
with 3-5x less error when activations contain outlier channels. `exact` stays within 0.2% relative error where per-token kernels
reach 7-13%.

## Requirements

- NVIDIA GPU with compute capability 8.0 or newer (RTX 30XX, 40XX, 50XX, A100, H100, B200 and later), Windows or Linux x86_64.
- PyTorch with CUDA, Python 3.10 or newer.
- Driver: R570 or newer for the CUDA 12.8 builds (`+cu128`), R580 or newer for the CUDA 13 builds (`+cu130`).
- Linux: glibc 2.34 or newer (Ubuntu 22.04, Debian 12, RHEL 9 or later).

Native code is included for SM80, 86, 89, 90, 100 and 120, plus PTX that newer GPUs compile on first use. Validation was performed on
an RTX 5090; on other GPUs, run `selftest()` (WanGP does it before selecting the kernels).

**AMD (experimental)**: a HIP build for RDNA4 (RX 9070 series, `gfx1201`) on Windows with ROCm PyTorch 2.10.0+rocm7.14.0. The INT8 GEMM
uses rocWMMA (row, `exact` and `block` modes as whole tiles, without the split-K schedule) and the few-token paths are CUDA-core GEMVs
with FP32 accumulation, so precision matches the NVIDIA paths. It was compiled and loaded without an AMD GPU: its kernels have not run
on AMD hardware yet, so `selftest()` decides whether they can be used.

## Installation

Pick the wheel matching your platform and CUDA stack. A wheel serves every Python and PyTorch version of its platform.

```bash
# Windows, PyTorch with CUDA 13 (e.g. PyTorch 2.10)
pip install https://github.com/deepbeepmeep/kernels/releases/download/wgpkernels-v1.0.0/wgpkernels-1.0.0%2Bcu130-py3-none-win_amd64.whl
# Windows, PyTorch with CUDA 12.8 (e.g. PyTorch 2.7.1)
pip install https://github.com/deepbeepmeep/kernels/releases/download/wgpkernels-v1.0.0/wgpkernels-1.0.0%2Bcu128-py3-none-win_amd64.whl
# Linux, PyTorch with CUDA 13
pip install https://github.com/deepbeepmeep/kernels/releases/download/wgpkernels-v1.0.0/wgpkernels-1.0.0%2Bcu130-py3-none-linux_x86_64.whl
# Linux, PyTorch with CUDA 12.8
pip install https://github.com/deepbeepmeep/kernels/releases/download/wgpkernels-v1.0.0/wgpkernels-1.0.0%2Bcu128-py3-none-linux_x86_64.whl
# Windows, AMD RX 9070 with PyTorch 2.10.0+rocm7.14.0 (experimental)
pip install https://github.com/deepbeepmeep/kernels/releases/download/wgpkernels-v1.0.0/wgpkernels-1.0.0%2Brocm714-py3-none-win_amd64.whl
```

## Usage

```python
import torch
from wgpkernels import int8 as wq

wq.load_library()          # the library bundled with the package
assert wq.selftest() is None

x = torch.randn(4096, 5120, dtype=torch.bfloat16, device="cuda")
weight = torch.randint(-127, 128, (3072, 5120), dtype=torch.int8, device="cuda")  # INT8 (N, K)
scale = torch.rand(3072, device="cuda") / 127                                      # FP32 per output channel
norm_weight = torch.ones(5120, dtype=torch.bfloat16, device="cuda")

y = wq.linear(x, weight, scale)                       # automatic path
y = wq.linear(x, weight, scale, precise=True)         # full precision at every length
y = wq.linear(x, weight, scale, act="rms_norm", act_weight=norm_weight, eps=1e-6)  # fused RMSNorm
y = wq.linear(x, weight, scale, group=256)            # ConvRot checkpoint (rotation group 16, 64 or 256)

prepared = wq.prepare(x, path=wq.choose_path(x, weight, scale))   # one quantization for several projections
q = wq.linear_prepared(prepared, weight, scale)
```

Activations are 2-D BF16 or FP16 tensors with contiguous rows aligned on 16 bytes; K must be a multiple of 16. Calls are compatible with
CUDA graph capture once a shape has run eagerly (its tuning is then cached); during capture or `torch.compile` tracing, fixed
thresholds replace the timing.

## Building from source

Requirements: CUDA toolkit 12.8 or newer, and MSVC (Windows, found automatically through `vswhere`) or gcc (Linux).

```bash
python -m wgpkernels.build                       # all architectures (80,86,89,90,100,120) into wgpkernels/int8/
python -m wgpkernels.build --arch 120 --verbose  # one architecture, with register and spill report
python -m wgpkernels.build --hip <ROCm SDK root> # AMD gfx1201 with the SDK's hipcc (e.g. ROCm 7.14's _rocm_sdk_devel)
pip install -e .                                 # editable install using the library built in place
```

Wheels are platform-tagged (`py3-none-<platform>`); `WGPKERNELS_VARIANT` sets the local version label naming the CUDA toolkit:

```bash
WGPKERNELS_VARIANT=cu130 python -m pip wheel . --no-deps --no-build-isolation -w dist
```

## License

Apache License 2.0, see [LICENSE.txt](LICENSE.txt).
