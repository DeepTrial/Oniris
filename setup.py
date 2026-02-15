"""Setup script for Oniris - ONNX Compilation Toolkit."""

import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """CMake extension module."""
    
    def __init__(self, name: str, sourcedir: str = ""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """CMake build command."""
    
    def run(self):
        """Run CMake build."""
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build the extension")
        
        for ext in self.extensions:
            self.build_extension(ext)
    
    def build_extension(self, ext):
        """Build a single extension."""
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep
        
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DBUILD_PYTHON_BINDINGS=ON",
        ]
        
        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]
        
        if sys.platform == "win32":
            cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
        
        cmake_args += [f"-DCMAKE_BUILD_TYPE={cfg}"]
        build_args += ["--", "-j4"]
        
        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)
        
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_temp)


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="oniris",
    version="0.1.0",
    author="Oniris Team",
    description="ONNX Compilation Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[CMakeExtension("oniris._oniris", sourcedir=".")],
    cmdclass={"build_ext": CMakeBuild},
    install_requires=[
        "numpy>=1.20.0",
        "onnx>=1.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "mypy>=1.0.0",
        ],
        "test": [
            "transformers>=4.20.0",
            "torch>=1.12.0",
        ],
    },
    python_requires=">=3.8",
    zip_safe=False,
)
