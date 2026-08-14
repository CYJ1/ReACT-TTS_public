"""Build Cython extension for monotonic alignment."""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext = Extension(
    "react_tts.tts.grad_tts.monotonic_align.core",
    sources=["react_tts/tts/grad_tts/monotonic_align/core.pyx"],
    include_dirs=[numpy.get_include()],
)

setup(
    name="monotonic_align",
    ext_modules=cythonize([ext], language_level=3),
)
