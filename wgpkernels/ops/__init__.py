# Copyright (c) 2026 DeepBeepMeep - Apache License 2.0 (see LICENSE.txt)
"""CUDA AdaLN, PixelNorm + SiLU and FP32-accumulating valid Conv3D.

Inference-only, FP16/BF16, CUDA SM80+. No tensor caches, compilation or provider
fallback. Callers select this implementation only for supported operands.
"""
import ctypes
import functools
import os
from pathlib import Path
import torch
from . import tuning

_lib = None
_tunable = False
_DTYPES = {torch.float16: 0, torch.bfloat16: 1}


def default_library_path():
    current = Path(__file__).with_name('wgp_ops_v2.dll' if os.name == 'nt' else 'libwgp_ops_v2.so')
    return current if current.exists() else Path(__file__).with_name('wgp_ops.dll' if os.name == 'nt' else 'libwgp_ops.so')


def load_library(path=None):
    global _lib, _tunable
    lib = ctypes.CDLL(str(Path(path or default_library_path()).resolve()))
    if lib.wops_abi() != 1:
        raise RuntimeError('Activation library ABI mismatch; rebuild it')
    p, i, q, f = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_float
    lib.wops_adaln.argtypes = [p,p,p,p,i,q,i,i,q,q,q,q,f,p]
    lib.wops_pixel.argtypes = [p,p,i,q,i,i,i,i,q,q,q,q,q,f,p]
    lib.wops_conv.argtypes = [p,p,p,p,i,q,i,i,i,i,i,i,i,i,i,i,i,p]
    _tunable = hasattr(lib, 'wops_tuning_version') and lib.wops_tuning_version() == tuning.VERSION
    if _tunable:
        lib.wops_adaln_config.argtypes = lib.wops_adaln.argtypes + [i]
        lib.wops_pixel_config.argtypes = lib.wops_pixel.argtypes + [i]
        lib.wops_conv_config.argtypes = lib.wops_conv.argtypes + [i]
    _lib = lib
    tuning._plans.clear()  # Explicit library reload cannot retain the old plans.
    return lib


@functools.cache
def supported(device):
    return device.type == 'cuda' and torch.version.hip is None and torch.cuda.get_device_capability(device) >= (8, 0)


def _check(code):
    if code:
        raise RuntimeError(f'Activation kernel launch failed (CUDA error {code})')


def _stream(x):
    return torch._C._cuda_getCurrentRawStream(x.device.index)


def _validate(x):
    if x.dtype not in _DTYPES or not supported(x.device) or torch.is_grad_enabled():
        raise ValueError('Expected inference-only FP16/BF16 CUDA tensors on SM80+')


def modulation_strides(t, x):
    """Broadcast metadata only; never expand/copy modulation tensors."""
    if t.device != x.device or t.dtype != x.dtype or t.ndim > x.ndim or t.shape[-1] != x.shape[-1] or t.stride(-1) != 1:
        raise ValueError('Modulation must broadcast to x, with matching dtype/device and contiguous channels')
    v = t.expand(x.shape)
    return (0, v.stride(0)) if x.ndim == 2 else v.stride()[:2]


def _launch(name, args, identity, device, candidates, config):
    lib = _lib or load_library()
    if not _tunable:
        if config not in (None, 0):
            raise ValueError('This activation library has no configurable kernels; rebuild it')
        _check(getattr(lib, 'wops_' + name)(*args))
        return
    function = getattr(lib, 'wops_' + name + '_config')
    def launch(choice):
        _check(function(*args, choice))
    if config is None:
        config = tuning.choose(name, identity, device.index, candidates, launch)
    elif config not in candidates:
        raise ValueError(f'Unsupported {name} configuration {config}')
    launch(config)


def adaln(x, scale, shift, eps=1e-6, *, config=None):
    """LayerNorm(x) * (1 + scale) + shift, with native dtype rounding points."""
    _validate(x)
    if x.ndim not in (2, 3) or not x.is_contiguous():
        raise ValueError('AdaLN expects contiguous rows in a 2-D or 3-D tensor')
    s0,s1 = modulation_strides(scale,x)
    h0,h1 = modulation_strides(shift,x)
    y = torch.empty_like(x)
    args = (x.data_ptr(),scale.data_ptr(),shift.data_ptr(),y.data_ptr(),_DTYPES[x.dtype],x.numel()//x.shape[-1],x.shape[-1],x.shape[-2],s0,s1,h0,h1,eps,_stream(x))
    _launch('adaln', args, (tuple(x.shape), _DTYPES[x.dtype], s0,s1,h0,h1), x.device, (0,1,2), config)
    return y


def pixel_norm_silu(x, eps=1e-6, *, config=None):
    """Channel PixelNorm followed by SiLU, written in channels-last-3d layout."""
    _validate(x)
    if x.ndim != 5:
        raise ValueError('PixelNorm expects B,C,T,H,W')
    b,c,t,h,w = x.shape
    y = torch.empty_like(x,memory_format=torch.channels_last_3d)
    candidates = (0,)
    if c in (32,64,128,256,512,1024) and b*t*h*w <= 2147483616 and t*h*w <= 2147483647:
        # Limit live registers per thread; this is a kernel resource constraint.
        candidates += tuple(index for index, lanes in enumerate((4,8,16,32),1) if c//lanes <= 64)
    args = (x.data_ptr(),y.data_ptr(),_DTYPES[x.dtype],b*t*h*w,c,t,h,w,*x.stride(),eps,_stream(x))
    _launch('pixel', args, (tuple(x.shape),tuple(x.stride()),_DTYPES[x.dtype]), x.device, candidates, config)
    return y


def conv3d(x, weight, bias=None, *, config=None):
    """Valid stride-one convolution. FP32 accumulation; no full im2col buffer."""
    _validate(x)
    if x.ndim!=5 or weight.ndim!=5 or x.shape[1]!=weight.shape[1] or x.shape[1]%32 or weight.shape[0]%32:
        raise ValueError('Conv3D expects 5-D tensors with input/output channels divisible by 32')
    if weight.device!=x.device or weight.dtype!=x.dtype or (bias is not None and (bias.device!=x.device or bias.dtype!=x.dtype or bias.shape!=(weight.shape[0],) or not bias.is_contiguous())):
        raise ValueError('Convolution parameters must match activation dtype/device')
    x = x.contiguous(memory_format=torch.channels_last_3d)
    weight = weight.contiguous(memory_format=torch.channels_last_3d)
    b,cin,tp,hp,wp = x.shape
    cout,_,kt,kh,kw = weight.shape
    t,h,w = tp-kt+1,hp-kh+1,wp-kw+1
    if min(t,h,w)<=0:
        raise ValueError('Kernel larger than convolution input')
    y = torch.empty((b,cout,t,h,w),device=x.device,dtype=x.dtype,memory_format=torch.channels_last_3d)
    candidates = (0,1,2,3) if kw*cin%64 == 0 else (0,1)
    args = (x.data_ptr(),weight.data_ptr(),None if bias is None else bias.data_ptr(),y.data_ptr(),_DTYPES[x.dtype],b*t*h*w,cin,cout,t,h,w,tp,hp,wp,kt,kh,kw,_stream(x))
    _launch('conv', args, (tuple(x.shape),tuple(weight.shape),_DTYPES[x.dtype],bias is not None), x.device, candidates, config)
    return y
