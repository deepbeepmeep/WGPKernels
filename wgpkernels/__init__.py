# Copyright (c) 2026 DeepBeepMeep - Apache License 2.0 (see LICENSE.txt)
"""GPU kernels for WanGP, built ahead of time into C ABI libraries loaded with ctypes (no dependency on a PyTorch or CUDA toolkit
version at run time; PyTorch supplies the tensors and the current stream).

Kernel families:
- wgpkernels.int8: INT8 weight linears (Quanto INT8 and ConvRot checkpoints).
- wgpkernels.ops: AdaLN, PixelNorm + SiLU, FP32-accumulating valid Conv3D (CUDA).
"""
__version__ = "1.1.0"
