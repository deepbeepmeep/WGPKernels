# Copyright (c) 2026 DeepBeepMeep - Apache License 2.0 (see LICENSE.txt)
# With ideas of Strata: the fused int8 prompt experts (MIT License, https://github.com/Niko1221/Strata).
"""Routed experts (SwiGLU) of a pass of many tokens on int8 tensor-core kernels, the weights read in their GGUF storage: activations and the SwiGLU's output as int8 with a float32 scale per 32 values, each dot an exact integer.
ExpertPass groups the pairs (token, slot) by expert on the host; run() computes experts as their rows arrive, finish() gives the float32 [tokens, hidden] sum. Launches go on PyTorch's current stream: no host wait.
forward_window() is the same for a decoding window, inside CUDA graphs: the grouping is a kernel, and activations and the SwiGLU's output take two int8 per value (16 bits of precision against the maximum of 32 values, relative error about 4e-5)."""
import ctypes
import functools
import os
from pathlib import Path

import numpy as np
import torch

IQ3_XXS, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, Q2_0 = 18, 20, 21, 22, 23, 42  # ggml types
GATE_UP_BLOCKS = {IQ2_S: (82, 256), IQ3_XXS: (98, 256), IQ3_S: (110, 256), IQ4_XS: (136, 256), Q2_0: (18, 64)}  # bytes per block, values per block
DOWN_BLOCKS = {IQ4_NL: (18, 32), Q2_0: (18, 64)}
WIDE_TILE_SMS = 80  # measured on 170 multiprocessors, the formats of a checkpoint at 1K, 4K and 8K tokens: 128 rows are the fastest of 64, 128 and 256, up to 20 % faster than 64 (at 48, Strata measured 64 faster at 8K)
_lib = None


def default_library_path():
    return Path(__file__).with_name("wgp_experts.dll" if os.name == "nt" else "libwgp_experts.so")


def load_library(path=None):
    global _lib
    lib = ctypes.CDLL(str(Path(path or default_library_path()).resolve()))
    if lib.wgx_abi() != 1:
        raise RuntimeError("Experts library ABI mismatch; rebuild it")
    p, i, i64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64
    lib.wgx_blocks_per_sm.argtypes = [i, i, i, i]
    lib.wgx_quantize.argtypes = [p, i64, i, p, p]
    lib.wgx_gate_up.argtypes = [i, i, i, p, p, p, p, p, i, i, i, i, i, i, p]
    lib.wgx_down.argtypes = [i, i, i, p, p, p, p, p, p, i, i, i, i, i, p]
    if hasattr(lib, "wgx_window_forward"):
        lib.wgx_window_scratch.argtypes = [i, i, i, i]
        lib.wgx_window_scratch.restype = i64
        lib.wgx_window_forward.argtypes = [i, i, i, p, i, p, i, p, i, p, i, i, i, i, i, p, p, p]
    _lib = lib
    return lib


def _library():
    return _lib or load_library()


def _check(code):
    if code:
        raise RuntimeError(f"Experts kernel launch failed (CUDA error {code})")


@functools.cache
def supported(device, gate_type, down_type):
    """The library is there, compute capability 8.0 or newer, both formats covered."""
    return device.type == "cuda" and torch.version.hip is None and torch.cuda.get_device_capability(device) >= (8, 0) and gate_type in GATE_UP_BLOCKS and down_type in DOWN_BLOCKS and default_library_path().exists()


@functools.cache
def _blocks(device_index, type, gu, ww):
    """Blocks of a launch on this GPU: as many as run at once."""
    per_sm = _library().wgx_blocks_per_sm(device_index, type, gu, ww)
    if per_sm < 1:
        raise RuntimeError(f"The experts kernel of type {type} does not run on this GPU")
    return per_sm * torch.cuda.get_device_properties(device_index).multi_processor_count


def groups(ids, rows=None, order=None, sms=0):
    """On the host: (pairs int32 [routed rows]: the pair of each row, the rows of an expert one after the other; tiles int32 [tiles, 2]: {expert | rows << 16, first row}; first int64 [len(order) + 1]: the first tile of each expert in order; rows).
    ids: int [tokens, topk]; order: the routed experts in their order of processing, by default ascending; rows: routed rows per tile, by default 128 on a GPU of WIDE_TILE_SMS multiprocessors or more, else when an expert has about one such tile (56 to 112 rows on average), else 64."""
    flat = ids.reshape(-1).numpy().astype(np.uint16)
    sorted_pairs = np.argsort(flat, kind="stable").astype(np.int32)  # radix sort
    counts = np.bincount(flat)
    if order is None:
        order = np.nonzero(counts)[0]
    if rows is None:
        rows = 128 if sms >= WIDE_TILE_SMS or 56 * len(order) < len(flat) <= 112 * len(order) else 64
    starts = np.zeros(len(counts) + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts)
    lengths = counts[order]
    needed = (lengths + rows - 1) // rows
    first = np.zeros(len(order) + 1, dtype=np.int64)
    first[1:] = np.cumsum(needed)
    offsets = np.zeros(len(order) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    pairs = np.concatenate([sorted_pairs[starts[expert]:starts[expert + 1]] for expert in order.tolist()]) if len(order) else sorted_pairs[:0]
    within = np.arange(first[-1]) - np.repeat(first[:-1], needed)  # tile within its expert
    tiles = np.empty((int(first[-1]), 2), dtype=np.int32)
    tiles[:, 0] = np.repeat(order, needed) | (np.minimum(rows, np.repeat(lengths, needed) - within * rows) << 16)
    tiles[:, 1] = np.repeat(offsets[:-1], needed) + within * rows
    return pairs, tiles, first, rows


def row_bytes(type, columns, table):
    bytes_per_block, values = table[type]
    if columns % values:
        raise ValueError(f"{columns} columns are not a whole number of blocks of type {type}")
    return columns // values * bytes_per_block


def check_pointers(pointers, experts):
    """Debug: reads the pointer table on the host (a sync): the experts about to run have addresses, 2-byte aligned."""
    rows = pointers[torch.as_tensor(experts, device=pointers.device)].to("cpu")
    if not bool(((rows != 0) & (rows % 2 == 0)).all()):
        raise ValueError("An expert has no address or a misaligned one")


class ExpertPass:
    """x: BF16 [tokens, hidden] on the GPU; ids: int [tokens, topk] on the host; probabilities [tokens, topk] on the GPU; types: (gate, up, down) ggml types; order: the routed experts in their order of processing. rows, blocks: launch configuration."""
    def __init__(self, x, ids, probabilities, intermediate, types, order=None, zero=False, rows=None, blocks=None):
        self.tokens, self.hidden = x.shape
        self.topk = ids.shape[-1]
        self.intermediate = intermediate
        self.gate, up, self.down = types
        device = x.device
        if self.gate != up or not supported(device, self.gate, self.down):
            raise ValueError(f"The experts kernels do not take the types {types} on {device}")
        if x.dtype != torch.bfloat16 or not x.is_contiguous() or ids.is_cuda or not probabilities.is_cuda or probabilities.device != device:
            raise ValueError("The experts kernels take contiguous BF16 activations and probabilities on the GPU, and the routing on the host")
        if self.hidden % 256 or intermediate % 128:
            raise ValueError(f"The experts kernels take 256 hidden columns and 128 intermediate columns at a time, not {self.hidden} and {intermediate}")
        self.probabilities = probabilities.reshape(-1).float().contiguous()
        if rows not in (None, 64, 128):
            raise ValueError("A tile has 64 or 128 routed rows")
        pairs, tiles, self.first, self.rows = groups(ids, rows, None if order is None else np.asarray(order, dtype=np.int64), torch.cuda.get_device_properties(device).multi_processor_count)
        self.ww = 256 // self.rows
        self.up_bytes, self.down_bytes = row_bytes(self.gate, self.hidden, GATE_UP_BLOCKS), row_bytes(self.down, intermediate, DOWN_BLOCKS)
        self.up_blocks = blocks or _blocks(device.index, self.gate, 1, self.ww)
        self.down_blocks = blocks or _blocks(device.index, self.down, 0, self.ww)
        self.pairs, self.tiles = torch.from_numpy(pairs).to(device), torch.from_numpy(tiles).to(device)
        self.codes = torch.empty(self.tokens * (self.hidden // 64) * 80, dtype=torch.uint8, device=device)
        self.mid = torch.empty(len(pairs) * (intermediate // 64) * 80, dtype=torch.uint8, device=device)
        self.parts = (torch.zeros if zero else torch.empty)(self.tokens * self.topk, self.hidden, dtype=torch.float32, device=device)
        for tensor in (x, self.codes, self.mid, self.parts, self.pairs, self.tiles, self.probabilities):
            if not tensor.is_cuda or tensor.data_ptr() % 2:
                raise ValueError("A misaligned or non-GPU buffer")
        if self.codes.data_ptr() % 16 or self.mid.data_ptr() % 16 or self.parts.data_ptr() % 4:
            raise ValueError("A misaligned kernel buffer")
        self.quantize(x)

    def quantize(self, x):
        _check(_library().wgx_quantize(x.data_ptr(), self.tokens, self.hidden, self.codes.data_ptr(), torch._C._cuda_getCurrentRawStream(x.device.index)))

    def run(self, pointers, start=0, stop=None, parts="both"):
        """The experts [start, stop) of the order of processing, whose rows pointers gives: int64 [experts, 3] on the GPU, the addresses of the gate, up and down rows of each."""
        first = int(self.first[start])
        count = int(self.first[len(self.first) - 1 if stop is None else stop]) - first
        if not count:
            return
        if not pointers.is_cuda or pointers.dtype != torch.int64 or pointers.ndim != 2 or pointers.shape[1] != 3 or not pointers.is_contiguous():
            raise ValueError("The pointer table must be a contiguous int64 [experts, 3] tensor on the GPU")
        stream = torch._C._cuda_getCurrentRawStream(self.codes.device.index)
        lib = _library()
        if parts != "down":
            _check(lib.wgx_gate_up(self.gate, self.ww, self.up_blocks, pointers.data_ptr(), self.tiles.data_ptr(), self.pairs.data_ptr(), self.codes.data_ptr(), self.mid.data_ptr(), first, count, self.hidden, self.intermediate, self.topk, self.up_bytes, stream))
        if parts != "up":
            _check(lib.wgx_down(self.down, self.ww, self.down_blocks, pointers.data_ptr(), self.tiles.data_ptr(), self.pairs.data_ptr(), self.probabilities.data_ptr(), self.mid.data_ptr(), self.parts.data_ptr(), first, count, self.hidden, self.intermediate, self.down_bytes, stream))

    def finish(self):
        self.codes = self.mid = self.pairs = self.tiles = self.probabilities = None
        output = self.parts.view(self.tokens, self.topk, self.hidden).sum(1)
        self.parts = None
        return output


WINDOW_MAX_TOKENS = 8
WINDOW_MAX_PAIRS = 128


@functools.cache
def window_supported(device, gate, down):
    """The kernels of a decoding window are in the library, on a GPU that has them, for both formats."""
    return supported(device, gate, down) and hasattr(_library(), "wgx_window_forward")


def window_scratch_bytes(tokens, topk, hidden, intermediate):
    """Bytes of the scratch of forward_window: a preallocated uint8 tensor, 16-byte aligned (at 5 tokens, 10 routes, 2560 and 640 columns: 113 KB)."""
    return _library().wgx_window_scratch(tokens, topk, hidden, intermediate)


def forward_window(x, ids, probabilities, pointers, intermediate, types, out, scratch, parts=7):
    """The experts of a window of 1 to 8 tokens into out, on the current stream, with no read on the host and no allocation: replayed in a CUDA graph, any routing and any change of the pointer table (both read on the GPU) gives the right result. Grids depend on tokens and topk only.
    x: contiguous BF16 or float32 [tokens, hidden]; ids: int32 or int64 [tokens, topk], contiguous; probabilities: float32 or BF16 [tokens, topk], contiguous; pointers: int64 [experts, 3], the addresses of the gate, up and down rows of each expert, zero for those without a GPU copy, which contribute zero;
    out: float32 [tokens, hidden], every row written; scratch: window_scratch_bytes(...) bytes. types: (gate, up, down) ggml types. parts: 1 the plan, 2 gate/up, 4 down."""
    tokens, hidden = x.shape
    topk = ids.shape[-1]
    gate, up, down = types
    if gate != up or not window_supported(x.device, gate, down):
        raise ValueError(f"The window kernels do not take the types {types} on {x.device}")
    if not (x.is_cuda and ids.is_cuda and probabilities.is_cuda and pointers.is_cuda and out.is_cuda and scratch.is_cuda):
        raise ValueError("The window kernels read GPU memory only")
    if tokens > WINDOW_MAX_TOKENS or tokens * topk > WINDOW_MAX_PAIRS or hidden % 256 or intermediate % 128:
        raise ValueError(f"A window of {tokens} tokens, {topk} routes, {hidden} hidden and {intermediate} intermediate columns")
    if x.dtype not in (torch.bfloat16, torch.float32) or ids.dtype not in (torch.int32, torch.int64) or probabilities.dtype not in (torch.float32, torch.bfloat16) or out.dtype != torch.float32 or pointers.dtype != torch.int64:
        raise ValueError("The window kernels take BF16 or float32 activations, int32 or int64 ids, float32 or BF16 probabilities, int64 pointers and a float32 output")
    if not (x.is_contiguous() and ids.is_contiguous() and probabilities.is_contiguous() and pointers.is_contiguous() and out.is_contiguous()) or pointers.shape[1] != 3 or tuple(out.shape) != (tokens, hidden) or ids.numel() != probabilities.numel():
        raise ValueError("The window kernels take contiguous tensors of matching shapes")
    if x.data_ptr() % x.element_size() or scratch.data_ptr() % 16 or out.data_ptr() % 4 or ids.data_ptr() % 4 or pointers.data_ptr() % 8 or scratch.numel() * scratch.element_size() < window_scratch_bytes(tokens, topk, hidden, intermediate):
        raise ValueError("A misaligned or too small window buffer")
    _check(_library().wgx_window_forward(gate, down, parts, x.data_ptr(), x.dtype == torch.float32, ids.data_ptr(), ids.element_size(), probabilities.data_ptr(), probabilities.dtype == torch.bfloat16, pointers.data_ptr(), pointers.shape[0], tokens, topk, hidden, intermediate, out.data_ptr(), scratch.data_ptr(), torch._C._cuda_getCurrentRawStream(x.device.index)))
