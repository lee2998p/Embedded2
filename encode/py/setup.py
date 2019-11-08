#!/usr/bin/env python3
from setuptools import setup, Extension
from torch.utils import cpp_extension
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import subprocess
import sys
import platform


if platform.system() == "Darwin":
    print("Mac OS systems not supported")
    sys.exit(-1)


# TODO add building NvPipe automagically

setup(
        name='video_encode',
        ext_modules=[
            CUDAExtension('video_encode', ['video_encode.cpp'],
                                        extra_compile_args = ['-Wall','-g', '-O0'],
                                        libraries = ['NvPipe']
            )
        ],
        cmdclass={'build_ext': BuildExtension},
        include_dirs = ['/usr/local/cuda/include'],
        install_requires=['torch>=1.3']
     )
