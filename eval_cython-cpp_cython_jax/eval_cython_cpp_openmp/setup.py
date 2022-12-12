import numpy as np
from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize('ucb_score_cython_cpp_openmp.pyx'), extra_compile_args=['-O3', '-fopt-info','-fopenmp', '-mavx2'], include_dirs=[np.get_include()])