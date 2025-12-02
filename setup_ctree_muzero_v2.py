"""
Setup script for building ctree_muzero_v2 Cython extension only.
This is a minimal setup focused on compiling the mz_tree.pyx module.
"""
import os
import sys
from distutils.core import setup

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension
from distutils.sysconfig import get_python_inc

# Get the directory of this script
here = os.path.abspath(os.path.dirname(__file__))

# Configure Python and include directories
python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
python_include_dir = get_python_inc()
include_dirs = [np.get_include(), python_include_dir]

# Add macOS homebrew path if available
if sys.platform == 'darwin':
    homebrew_python_path = f'/usr/local/opt/python@{python_version}/Frameworks/Python.framework/Versions/{python_version}/include/python{python_version}'
    if os.path.exists(homebrew_python_path):
        include_dirs.append(homebrew_python_path)

print(f"Python version: {python_version}")
print(f"Include directories: {include_dirs}")

# Set C++11 compile parameters according to the operating system
if sys.platform == 'win32':
    # Use the VS compiler on Windows platform
    extra_compile_args = ["/std:c++11"]
    extra_link_args = ["/std:c++11"]
else:
    # Linux/macOS Platform
    extra_compile_args = ["-std=c++11"]
    extra_link_args = ["-std=c++11"]

# Path to ctree_muzero_v2 directory
ctree_muzero_v2_path = os.path.join(here, 'lzero', 'mcts', 'ctree', 'ctree_muzero_v2')

# Create extension for mz_tree
mz_tree_pyx = os.path.join(ctree_muzero_v2_path, 'mz_tree.pyx')

if not os.path.exists(mz_tree_pyx):
    raise FileNotFoundError(f"Cannot find {mz_tree_pyx}")

# Build the extension module name: lzero.mcts.ctree.ctree_muzero_v2.mz_tree
ext_module = Extension(
    'lzero.mcts.ctree.ctree_muzero_v2.mz_tree',
    [mz_tree_pyx],
    include_dirs=include_dirs,
    language="c++",
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

# Check if LINETRACE environment variable is set
_LINETRACE = bool(os.environ.get('LINETRACE', None))

setup(
    name='lzero-ctree-muzero-v2',
    version='0.2.0',
    description='Cython extension for MuZero v2 MCTS tree.',
    author='opendilab',
    author_email='opendilab@pjlab.org.cn',
    url='https://github.com/opendilab/LightZero',
    license='Apache License, Version 2.0',
    py_modules=['lzero.mcts.ctree.ctree_muzero_v2'],
    python_requires=">=3.7",
    ext_modules=cythonize(
        [ext_module],
        language_level=3,
        compiler_directives=dict(
            linetrace=_LINETRACE,
        ),
    ),
)
