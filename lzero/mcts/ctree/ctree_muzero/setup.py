from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext

# Define the extension
ext = Extension("mz_tree",
                sources=["mz_tree.pyx", "lib/cnode.cpp"],
                language='c++',  # Note the language specification
                extra_compile_args=["-std=c++11"])

setup(
    name='mz_tree',
    cmdclass={'build_ext': build_ext},
    ext_modules=[ext],
)