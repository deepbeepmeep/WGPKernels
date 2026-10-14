# Copyright (c) 2026 DeepBeepMeep - Apache License 2.0 (see LICENSE.txt)
"""Platform wheels: the package ships a prebuilt kernel library loaded with ctypes (no Python extension, no PyTorch ABI), so a wheel
is tagged py3-none-<platform> and serves every Python and PyTorch version of its platform. WGPKERNELS_VARIANT (e.g. cu128) names the
CUDA toolkit of the bundled library as a local version label. Build the library first (python -m wgpkernels.build), then:
python -m pip wheel . --no-deps --no-build-isolation -w <dir>"""
import os
import re
from pathlib import Path

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel


class PlatformWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        return "py3", "none", super().get_tag()[2]


version = re.search(r'__version__ = "([^"]+)"', (Path(__file__).parent / "wgpkernels" / "__init__.py").read_text(encoding="utf-8")).group(1)
variant = os.environ.get("WGPKERNELS_VARIANT")
setup(version=f"{version}+{variant}" if variant else version, cmdclass={"bdist_wheel": PlatformWheel})
