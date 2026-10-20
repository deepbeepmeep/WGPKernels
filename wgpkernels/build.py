# Copyright (c) 2026 DeepBeepMeep - Apache License 2.0 (see LICENSE.txt)
"""Ahead-of-time build of the kernel libraries, written next to their sources:
python -m wgpkernels.build [--library int8] [--arch 120] [--output PATH] [--verbose]
AMD (HIP): python -m wgpkernels.build --hip <ROCm SDK root, holding bin/hipcc> [--arch gfx1201]"""
import argparse
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
# SASS for every listed architecture plus PTX for the newest, so later GPUs JIT once. Turing (sm_75) runs the INT8 kernels with synchronous copies and m8n8k16 MMAs;
# the activation / convolution library needs SM80.
DEFAULT_ARCHS = "all"
MIN_ARCH = {"int8": 75, "ops": 80}
DEFAULT_HIP_ARCHS = "gfx1200,gfx1201"  # RDNA4: the AMD targets need rocWMMA's gfx12 WMMA
# library name -> CUDA source, relative to the package
LIBRARIES = {"int8": "int8/int8_gemm.cu", "ops": "ops/ops.cu"}


def library_file(name):
    source = PACKAGE / LIBRARIES[name]
    binary_name = 'ops_v2' if name == 'ops' else name
    return source.with_name(f"wgp_{binary_name}.dll" if os.name == "nt" else f"libwgp_{binary_name}.so")


def _msvc_environment():
    env = os.environ.copy()
    if shutil.which("cl"):
        return env
    program_files = Path(os.environ["ProgramFiles(x86)"])
    # the newest Visual Studio that nvcc accepts (2019 to 2022): a newer one beside it fails nvcc's host compiler check
    install = Path(subprocess.check_output([str(program_files / "Microsoft Visual Studio/Installer/vswhere.exe"), "-latest", "-version", "[16.0,18.0)", "-products", "*",
                                            "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"], text=True).strip())
    version = lambda path: tuple(int(part) for part in path.name.split("."))
    vc = max((install / "VC/Tools/MSVC").iterdir(), key=version)
    sdk = program_files / "Windows Kits/10"
    sdk_version = max((sdk / "Include").iterdir(), key=version).name
    env["PATH"] = f"{vc / 'bin/Hostx64/x64'}{os.pathsep}{env['PATH']}"
    env["INCLUDE"] = ";".join(str(p) for p in [vc / "include", *(sdk / "Include" / sdk_version / part for part in ("ucrt", "shared", "um"))])
    env["LIB"] = ";".join(str(p) for p in [vc / "lib/x64", *(sdk / "Lib" / sdk_version / part / "x64" for part in ("ucrt", "um"))])
    return env


def build(name="int8", archs=DEFAULT_ARCHS, output=None, nvcc="nvcc", verbose=False):
    if archs == "all":
        codes = subprocess.check_output([nvcc, "--list-gpu-code"], text=True)
        archs = ",".join(code for code in re.findall(r"^sm_(\d+)$", codes, re.MULTILINE) if int(code) >= MIN_ARCH[name])
    archs = sorted({int(a) for a in str(archs).split(",")})
    target = Path(output or library_file(name)).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = [f"-gencode=arch=compute_{a},code=sm_{a}" for a in archs] + [f"-gencode=arch=compute_{archs[-1]},code=compute_{archs[-1]}"]
    # Static CUDA runtime and C runtime: the library depends on the driver only, not on a PyTorch or toolkit version.
    # Linux: hidden symbols except the API, so that no template static (GNU unique) is shared with another copy of the library.
    flags += ["-Xcompiler", "/MT" if os.name == "nt" else "-fPIC,-fvisibility=hidden", "--cudart", "static", "--threads", "0"]  # architectures in parallel
    flags += ["--compress-mode", "balance"]  # compressed device code, as CUDA 13 does by default (CUDA 12.8 otherwise doubles the size)
    if verbose:
        flags += ["-Xptxas", "-v"]
    env = _msvc_environment() if os.name == "nt" else os.environ.copy()
    with tempfile.TemporaryDirectory(prefix="wgpkernels-build-") as work:
        built = Path(work) / target.name
        subprocess.run([nvcc, "-shared", "-O3", "-std=c++17", "-lineinfo", *flags, str(PACKAGE / LIBRARIES[name]), "-o", str(built)], cwd=work, env=env, check=True)
        shutil.copyfile(built, target)
    return target


def build_hip(name="int8", archs=DEFAULT_HIP_ARCHS, output=None, rocm=None, verbose=False):
    """HIP build with the ROCm SDK's hipcc (one --offload-arch per GPU); the library links the HIP runtime that ROCm PyTorch loads."""
    if name == "ops":
        raise ValueError("The ops library uses NVIDIA WMMA; use the independent Triton/PyTorch operations on HIP")
    rocm = Path(rocm)
    target = Path(output or library_file(name)).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = [f"--offload-arch={arch}" for arch in str(archs).split(",")]
    flags += [f"--rocm-path={rocm}", f"--rocm-device-lib-path={rocm / 'lib/llvm/amdgcn/bitcode'}"] + ([] if os.name == "nt" else ["-fPIC", "-fvisibility=hidden"])
    flags += shlex.split(os.environ.get("WGPKERNELS_HIP_FLAGS", ""))  # e.g. an older glibc sysroot and its libstdc++ headers on Linux
    if verbose:
        flags += ["-Rpass-analysis=kernel-resource-usage"]
    env = _msvc_environment() if os.name == "nt" else os.environ.copy()
    env.update(HIP_PATH=str(rocm), ROCM_PATH=str(rocm), HIP_PLATFORM="amd")
    env["PATH"] = os.pathsep.join([str(rocm / "lib/llvm/bin"), str(rocm / "bin"), env["PATH"]])
    hipcc = rocm / "bin" / ("hipcc.exe" if os.name == "nt" else "hipcc")
    with tempfile.TemporaryDirectory(prefix="wgpkernels-build-") as work:
        built = Path(work) / target.name
        subprocess.run([str(hipcc), "-shared", "-O3", "-std=c++17", *flags, "-x", "hip", str(PACKAGE / LIBRARIES[name]), "-o", str(built)],
                       cwd=work, env=env, check=True)
        shutil.copyfile(built, target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", default="int8", choices=sorted(LIBRARIES))
    parser.add_argument("--arch", default=None, help="all supported targets (default: SM75+ for int8, SM80+ for ops), comma separated SM versions (CUDA), or gfx targets (HIP)")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--hip", default=None, help="ROCm SDK root: build for AMD GPUs with its hipcc")
    parser.add_argument("--verbose", action="store_true", help="print register and shared memory usage")
    args = parser.parse_args()
    if args.hip:
        print(f"Built {build_hip(args.library, args.arch or DEFAULT_HIP_ARCHS, args.output, args.hip, args.verbose)}")
    else:
        print(f"Built {build(args.library, args.arch or DEFAULT_ARCHS, args.output, args.nvcc, args.verbose)}")


if __name__ == "__main__":
    main()
