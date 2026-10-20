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

Autotuning prints a start message and the selected choice, without listing candidate timings or nested tile searches. Cache hits are
silent and survive restarts. INT8 path selection uses exact token counts through 16, groups of 8 through 128, then groups of 64 through
4096. GEMM tile selection uses groups of 64 through 4096, then buckets of 8192 and 16384; larger inputs reuse the 16384 bucket. Layer
widths remain distinct, and path choices also distinguish ConvRot groups, fused input operations and `precise=True`. In ordinary mode,
inputs above 4096 tokens use `block` directly. The first input in a bucket supplies the measurements reused for that bucket.

Fusions, available on every path and computed in FP32 with a single final rounding:
- input operation applied while the activation is quantized: `act="rms_norm"` (optional weight) or `act="swiglu"` (rows `[gate | value]`);
- residual epilogue: `Y = residual + residual_scale * (X @ W.T + bias)`, optionally written in place of the residual;
- `prepare()` quantizes an activation once for several weights or row ranges (`linear_prepared()`).

Measured on an RTX 5090 (whole linears, activation quantization included): `block` is about 1.65x faster than Triton INT8 kernels
with per-block activation scales at similar precision, and about 25% slower than per-token (W8A8) kernels on large matrix products,
with 3-5x less error when activations contain outlier channels. `exact` stays within 0.2% relative error where per-token kernels
reach 7-13%.

## Requirements

- NVIDIA GPU with compute capability 7.5 or newer for the INT8 kernels (RTX 20XX, GTX 16XX, T4, then RTX 30XX, 40XX, 50XX, A100, H100,
  B200 and later), 8.0 or newer for the activation / convolution kernels, Windows or Linux x86_64.
- PyTorch with CUDA, Python 3.10 or newer.
- Driver: R570 or newer for the CUDA 12.8 builds (`+cu128`), R580 or newer for the CUDA 13 builds (`+cu130`).
- Linux: glibc 2.34 or newer (Ubuntu 22.04, Debian 12, RHEL 9 or later).

Source builds include every base target supported by the selected CUDA toolkit (SM75+ for the INT8 library, SM80+ for the activation /
convolution library), plus PTX for its newest target. On Turing (SM75) the INT8 kernels use synchronous copies and m8n8k16 MMAs.
Validation was performed on an RTX 5090; on other GPUs, run `selftest()` (WanGP does it before selecting the INT8 kernels).

**AMD (experimental)**: a HIP build for RDNA4 (`gfx1200`, `gfx1201`) on Windows and Linux with PyTorch 2.13.0 / ROCm 10.0. The INT8 GEMM
uses rocWMMA (row, `exact` and `block` modes as whole tiles, without the split-K schedule); the few-token paths use GEMV with FP32
accumulation. Both wheels pass installation and library-loading checks. AMD kernel execution, numerical precision, speed, memory
usage and graph replay remain unvalidated, so `selftest()` decides whether the kernels can be used.

## Installation

Pick the wheel matching your platform and CUDA stack. A wheel serves every Python and PyTorch version of its platform.

```bash
# Windows, PyTorch with CUDA 13 (e.g. PyTorch 2.10)
pip install https://github.com/deepbeepmeep/WGPKernels/releases/download/v1.1.2/wgpkernels-1.1.2%2Bcu130-py3-none-win_amd64.whl
# Windows, PyTorch with CUDA 12.8 (e.g. PyTorch 2.7.1)
pip install https://github.com/deepbeepmeep/WGPKernels/releases/download/v1.1.2/wgpkernels-1.1.2%2Bcu128-py3-none-win_amd64.whl
# Linux, PyTorch with CUDA 13
pip install https://github.com/deepbeepmeep/WGPKernels/releases/download/v1.1.2/wgpkernels-1.1.2%2Bcu130-py3-none-linux_x86_64.whl
# Linux, PyTorch with CUDA 12.8
pip install https://github.com/deepbeepmeep/WGPKernels/releases/download/v1.1.2/wgpkernels-1.1.2%2Bcu128-py3-none-linux_x86_64.whl
# Windows, AMD RDNA4 with PyTorch 2.13.0 / ROCm 10.0 (experimental)
pip install --no-deps https://github.com/deepbeepmeep/WGPKernels/releases/download/v1.1.2/wgpkernels-1.1.2%2Brocm100-py3-none-win_amd64.whl
# Linux, AMD RDNA4 with PyTorch 2.13.0 / ROCm 10.0 (experimental)
pip install --no-deps https://github.com/deepbeepmeep/WGPKernels/releases/download/v1.1.2/wgpkernels-1.1.2%2Brocm100-py3-none-linux_x86_64.whl
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

### Activation and convolution library

`wgpkernels.ops` supplies independent CUDA implementations of AdaLN (LayerNorm
followed by scale and shift), channel PixelNorm followed by SiLU, and valid
stride-one Conv3D. It does not depend on Comfy Kitchen or Triton. Build it with
`python -m wgpkernels.build --library ops`; the INT8 library is built separately.

```python
from wgpkernels import ops
ops.load_library()
with torch.inference_mode():
    y = ops.adaln(x, scale, shift, eps=1e-6)
    # Five-dimensional B,C,T,H,W activations:
    y = ops.pixel_norm_silu(video, eps=1e-6)
    y = ops.conv3d(padded_video, weight, bias)
```

The operations accept FP16/BF16 CUDA tensors, use the current stream and support
CUDA graph capture after library loading. AdaLN accepts contiguous 2-D/3-D rows
and broadcast modulation with contiguous channels. PixelNorm accepts strided
inputs and returns channels-last-3d. Conv3D requires channel counts divisible by
32; it converts inputs/weights to channels-last when needed, uses implicit GEMM
without an im2col buffer, and returns channels-last-3d. No weights or activations
are cached. Callers own padding, temporal history and module hooks.

The configurable ops library (`wgp_ops_v2.dll` / `libwgp_ops_v2.so`) tunes native
CUDA configurations on first eager use, like the INT8 kernels. It compares
AdaLN block sizes, PixelNorm channel/thread groupings and Conv3D tiles without
changing providers. Measurements reuse the required output buffer; inputs and
weights remain read-only. Winners and timing metadata are saved in
`~/.triton/autotune/wgpkernels_ops.json` by GPU, dtype, geometry and layout.
`WGPKERNELS_OPS_CACHE` can select a different cache file. No GPU tensors, modules
or callbacks are kept in that cache. The cache version changes with the kernels,
candidate configurations or timing method.

Warm up before graph capture. A capture-time cache miss uses the baseline without
tuning or saving that temporary choice; a later eager call can still tune it.
Candidate configurations preserve the original reduction trees and rounding.
For diagnostics, `config=0` forces the original CUDA configuration. An older ops
library remains usable with its fixed configurations.

Statistics and convolution accumulate in FP32. AdaLN and PixelNorm retain their
PyTorch intermediate rounding points; parallel reduction order may differ.
Conv3D rounds its FP32 sum to the activation dtype before adding bias, matching
the established WanGP convolution epilogue. These APIs do not silently select
other providers. WanGP controls availability, precision eligibility and its
independent Triton/PyTorch fallbacks. The ops library uses NVIDIA WMMA and has no
HIP build; the existing INT8 HIP library is unchanged.

### Toolchain

Requirements: CUDA toolkit 12.8 or newer, and MSVC (Windows, found automatically through `vswhere`: the newest Visual Studio 2019-2022,
the versions nvcc accepts) or gcc (Linux).

```bash
python -m wgpkernels.build                       # every toolkit-supported SM75+ target into wgpkernels/int8/
python -m wgpkernels.build --library ops         # SM80+ targets for activation/convolution kernels
python -m wgpkernels.build --arch 120 --verbose  # one architecture, with register and spill report
python -m wgpkernels.build --hip <ROCm SDK root> # AMD gfx1200 and gfx1201 with ROCm 10's hipcc
pip install -e .                                 # editable install using the library built in place
```

Wheels are platform-tagged (`py3-none-<platform>`); `WGPKERNELS_VARIANT` sets the local version label naming the CUDA toolkit:

```bash
WGPKERNELS_VARIANT=cu130 python -m pip wheel . --no-deps --no-build-isolation -w dist
```

## License

Apache License 2.0, see [LICENSE.txt](LICENSE.txt).
