import numpy as np
from setuptools import setup
from Cython.Build import cythonize
setup(ext_modules=cythonize('cytree.pyx'), extra_compile_args=['-O3'], include_dirs=[np.get_include()])


# import os 
# os.environ['LDSHARED'] = 'clang -shared'


# setup(ext_modules=cythonize('cytree.pyx'), extra_compile_args=['-std=c++17'], include_dirs=[np.get_include()])

# setup(ext_modules=cythonize('cytree.pyx'), extra_compile_args=['-std=c++11'], include_dirs=[np.get_include()])

