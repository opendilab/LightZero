import numpy as np
from Cython.Build import cythonize
from setuptools import setup

setup(ext_modules=cythonize('cytree.pyx'), extra_compile_args=['-O3'], include_dirs=[np.get_include()])
