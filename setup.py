"""Build the optional native library into platform wheels when a compiler exists."""

import os

from setuptools import Extension, setup

compile_args = ["/std:c++17"] if os.name == "nt" else ["-std=c++17"]

setup(
    ext_modules=[
        Extension(
            "specdecode.libspecdecode_native",
            sources=["native/src/native.cpp"],
            include_dirs=["native/include"],
            define_macros=[("SD_NATIVE_BUILD", "1")],
            extra_compile_args=compile_args,
            language="c++",
            optional=True,
        )
    ]
)
