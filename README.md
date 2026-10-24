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

Group-quantized INT8 checkpoints, with one FP32 scale per output channel and 128-element block of K (`scale` of shape `(N, K / 128)`, for
example Hunyuan's group-128 INT8 DiT), run the `exact` and `block` paths: each block's exact INT32 sums are scaled by their weight scale
(and activation block scale) into FP32 accumulators. `linear()` takes such scales directly; the few-token paths need per-channel scales.

Autotuning prints a start message and the selected choice, without listing candidate timings or nested tile searches. Cache hits are
silent and survive restarts. INT8 path selection uses exact token counts through 16, groups of 8 through 128, then groups of 64 through
4096. GEMM tile selection uses groups of 64 through 4096, then buckets of 8192 and 16384; larger inputs reuse the 16384 bucket. Layer
widths remain distinct, and path choices also distinguish ConvRot groups, fused input operations and `precise=True`. In ordinary mode,
inputs above 4096 tokens use `block` directly. The first input in a bucket supplies the measurements reused for that bucket.

Fusions, available on every path and computed in FP32 with a single final rounding:
- input operation applied while the activation is quantized: `act="rms_norm"` (optional weight), `act="swiglu"` (rows `[gate | value]`) or `act="gelu"` (tanh approximation);
- residual epilogue: `Y = residual + residual_scale * (X @ W.T + bias)`, optionally written in place of the residual;
- `prepare()` quantizes an activation once for several weights or row ranges (`linear_prepared()`).

Measured on an RTX 5090 (whole linears, activation quantization included): `block` is about 1.65x faster than Triton INT8 kernels
with per-block activation scales at similar precision, and about 25% slower than per-token (W8A8) kernels on large matrix products,
with 3-5x less error when activations contain outlier channels. `exact` stays within 0.2% relative error where per-token kernels
reach 7-13%.

## Requirements

- NVIDIA GPU with compute capability 7.5 or newer for the INT8 and FP8 kernels (RTX 20XX, Quadro RTX, T4, then RTX 30XX, 40XX, 50XX, A100,
  H100, B200 and later; GTX 16XX, without tensor cores, is untested), 8.0 or newer for the activation / convolution kernels, Windows or Linux
  x86_64.
- PyTorch with CUDA, Python 3.10 or newer.
- Driver: R570 or newer for the CUDA 12.8 builds (`+cu128`), R580 or newer for the CUDA 13 builds (`+cu130`).
- Linux: glibc 2.34 or newer (Ubuntu 22.04, Debian 12, RHEL 9 or later).

Source builds include every base target supported by the selected CUDA toolkit (SM75+ for the INT8 and FP8 libraries, SM80+ for the
activation / convolution library), plus PTX for its newest target. On Turing (SM75) the INT8 kernels use synchronous copies and m8n8k16
MMAs.
Validation was performed on an RTX 5090; on other GPUs, run `selftest()` (WanGP does it before selecting the INT8 kernels).

**AMD (experimental)**: a HIP build for RDNA4 (`gfx1200`, `gfx1201`) on Windows and Linux with PyTorch 2.13.0 / ROCm 10.0. The INT8 GEMM
uses rocWMMA (row, `exact` and `block` modes as whole tiles, without the split-K schedule); the few-token paths use GEMV with FP32
accumulation; the FP8 GEMM uses rocWMMA's gfx12 FP8 WMMA. Both wheels pass installation and library-loading checks. AMD kernel execution,
numerical precision, speed, memory usage and graph replay remain unvalidated, so `selftest()` decides whether the kernels can be used.

## Installation

Pick the wheel matching your platform and CUDA stack. A wheel serves every Python and PyTorch version of its platform.

```bash
# Windows, PyTorch with CUDA 13 (e.g. PyTorch 2.10)
pip install https://github.com/deepbeepmeep/WGPKernels/releases/download/v1.4.0/wgpkernels-1.4.0%2Bcu130-py3-none-win_amd64.whl
# Windows, PyTorch with CUDA 12.8 (e.g. PyTorch 2.7.1)
pip install https://github.com/deepbeepmeep/WGPKernels/releases/download/v1.4.0/wgpkernels-1.4.0%2Bcu128-py3-none-win_amd64.whl
# Linux, PyTorch with CUDA 13
pip install https://github.com/deepbeepmeep/WGPKernels/releases/download/v1.4.0/wgpkernels-1.4.0%2Bcu130-py3-none-linux_x86_64.whl
# Linux, PyTorch with CUDA 12.8
pip install https://github.com/deepbeepmeep/WGPKernels/releases/download/v1.4.0/wgpkernels-1.4.0%2Bcu128-py3-none-linux_x86_64.whl
# Windows, AMD RDNA4 with PyTorch 2.13.0 / ROCm 10.0 (experimental)
pip install --no-deps https://github.com/deepbeepmeep/WGPKernels/releases/download/v1.4.0/wgpkernels-1.4.0%2Brocm100-py3-none-win_amd64.whl
# Linux, AMD RDNA4 with PyTorch 2.13.0 / ROCm 10.0 (experimental)
pip install --no-deps https://github.com/deepbeepmeep/WGPKernels/releases/download/v1.4.0/wgpkernels-1.4.0%2Brocm100-py3-none-linux_x86_64.whl
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

## FP8 block-scaled linears

`wgpkernels.fp8` runs FP8 (E4M3) checkpoints whose weights carry FP32 scales per 128 x 128 block (DeepGEMM's `per_block_cast_to_fp8`),
per output channel and 128-element block of K, per output channel or per tensor, with the semantics of DeepGEMM's FP8 GEMM and CUTLASS's
`GemmBlockwise`: each 128-element block of K
is accumulated from zero in FP32, then scaled by its activation and weight scales into FP32 accumulators. Activations are quantized to
E4M3 with one scale per token and 128-element block (`amax / 448`, as DeepGEMM's `per_token_cast_to_fp8`), or optionally one scale per
128 x 128 tile (the activation scaling of Hunyuan's sm89 CUTLASS path).

- RTX 40/50 (SM89, SM120): E4M3 tensor cores (on H100 and datacenter Blackwell, ptxas runs the same instruction on FP16 tensor cores).
  RTX 30, A100 (SM80-88) and Turing (SM75): E4M3 decoded exactly to FP16 in registers (FP16 tensor cores, FP32 accumulation). AMD RDNA4
  (experimental, kernel execution unvalidated): rocWMMA's gfx12 FP8 WMMA.
- Persistent tiles with stream-K; the results are deterministic. Bias and residual epilogue as for the INT8 kernels.
- These are approximate kernels: FP8 activations give about 2.5e-2 relative error per layer (about 1e-2 when activations have outlier
  channels), against about 2.5e-3 for FP8 weights with BF16 activations.
- RTX 4090, 8k-36k tokens at hidden size 4096 / MLP 12288: 296-312 TFLOPS for the GEMM, 1.60-1.76x the speed of a cuBLAS BF16 matmul
  including the activation quantization.

```python
from wgpkernels import fp8

fp8.load_library()
assert fp8.selftest() is None
y = fp8.linear(x, fp8_weight, fp8_scale)           # E4M3 (N, K) weight, FP32 (ceil(N / 128), ceil(K / 128)) scales
q, sa = fp8.quantize_blocks(x)                     # or quantize once for several weights...
y = fp8.gemm(q, sa, fp8_weight, fp8_scale, bias)   # ...and run the GEMM on the E4M3 activations
```

## Block-sparse attention (VSA)

`wgpkernels.attn` runs WanGP's video sparse attention (VSA): every 64-token query tile attends only the key tiles of its index list,
with INT8 Q / K (one scale per head and 64-row block) and partial key tiles masked by their real row count. It is derived from
SpargeAttn / SageAttention 2 (Apache 2.0, vendored in `wgpkernels/attn/sparge` with its license): FP8 P·V with per-channel V scales on
SM89+ and FP16 P·V on SM80-88. CUDA only (SM80+); WanGP uses its Triton path elsewhere.

```python
from wgpkernels import attn

attn.load_library()
kind = attn.kernel_kind(device)   # "fp8", "fp16" or None
attn.vsa_attention(q_int8, k_int8, v, out, lut, count, valid, q_scale, k_scale, v_scale, sm_scale)
```

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
python -m wgpkernels.build --library fp8         # SM75+ targets for the FP8 kernels (--hip <ROCm SDK root> for RDNA4)
python -m wgpkernels.build --library attn        # SM80+ targets for the VSA attention kernels (CUDA only)
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
