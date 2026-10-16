# Copyright (c) 2026 DeepBeepMeep - Apache License 2.0 (see LICENSE.txt)
"""Native CUDA launch tuning. Only configuration IDs and timings are cached.

Like the INT8 tuner, time candidates in alternating order on the real operands,
writing only the operation's disposable output. A queued delay keeps Python
launch latency out of GPU timings. Capture misses use the baseline without
caching it, leaving the next eager call free to tune.
"""
import functools
import json
import os
import threading
import time
from pathlib import Path

import torch

VERSION = 1  # Bump for kernel, candidates, key or benchmark changes.
MAX_ENTRIES = 4096
_path = Path(os.environ.get('WGPKERNELS_OPS_CACHE', Path.home() / '.triton' / 'autotune' / 'wgpkernels_ops.json'))
_entries = None
_plans = {}
_lock = threading.RLock()


@functools.cache
def _device_identity(index):
    props = torch.cuda.get_device_properties(index)
    return f'{props.name}|sm{props.major}{props.minor}|{props.multi_processor_count}|cuda{torch.version.cuda}|torch{torch.__version__}'


def _cache():
    global _entries
    if _entries is None:
        try:
            payload = json.loads(_path.read_text(encoding='utf-8'))
            _entries = payload['entries'] if payload['version'] == VERSION else {}
            if not isinstance(_entries, dict):
                _entries = {}
        except (OSError, ValueError, KeyError, TypeError):
            _entries = {}
    return _entries


def _save():
    while len(_entries) > MAX_ENTRIES:
        _entries.pop(next(iter(_entries)))
    temporary = _path.with_suffix(f'.{os.getpid()}.tmp')
    try:
        _path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps({'version': VERSION, 'entries': _entries}, indent=1), encoding='utf-8')
        temporary.replace(_path)
    except OSError as error:
        print(f'[Kernels] Could not save native CUDA tuning: {error}')
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def choose(operation, identity, device_index, candidates, launch):
    """launch(config) overwrites an output that is not an input or residual.

    The callback and operands are used only in this call, never stored. Identity
    includes dtype, geometry, layout and operand flags; never tensor addresses.
    """
    key = operation, identity, device_index, candidates
    if len(candidates) == 1:
        return candidates[0]
    cached = _plans.get(key)
    if cached is not None:
        return cached
    # Do not read/write files or synchronize during capture/compilation.
    blocked = torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing()
    if blocked:
        return candidates[0]
    with _lock:
        stored_key = repr((operation, identity, candidates, _device_identity(device_index)))
        entry = _cache().get(stored_key)
        if isinstance(entry, dict) and entry.get('config') in candidates:
            best = entry['config']
        else:
            started = time.perf_counter()
            print(f'[WGPKernels] Autotuning CUDA {operation}...', flush=True)
            timings = {config: float('inf') for config in candidates}
            for config in candidates:
                launch(config)
                launch(config)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for turn in range(3):
                for config in (candidates if turn % 2 == 0 else candidates[::-1]):
                    torch.cuda._sleep(1_000_000)
                    start.record()
                    for _ in range(5):
                        launch(config)
                    end.record()
                    end.synchronize()
                    timings[config] = min(timings[config], start.elapsed_time(end) / 5)
            # Baseline wins ties; don't change math for timing noise.
            best = min(candidates, key=timings.__getitem__)
            if timings[candidates[0]] <= timings[best] * 1.02:
                best = candidates[0]
            _entries[stored_key] = {'config': best, 'milliseconds': timings}
            _save()
            print(f'[WGPKernels] Selected CUDA {operation}: configuration {best} ({timings[best] * 1000:.1f} us; tuned in {time.perf_counter() - started:.2f} s).', flush=True)
        if len(_plans) >= MAX_ENTRIES:
            _plans.clear()
        _plans[key] = best
        return best
