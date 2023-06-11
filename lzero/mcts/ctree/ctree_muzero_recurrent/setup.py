from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("/Users/puyuan/code/LightZero/lzero/mcts/ctree/ctree_muzero_recurrent/mzr_tree.pyx", annotate=True),
    include_dirs=[np.get_include()],
)

# python /Users/puyuan/code/LightZero/lzero/mcts/ctree/ctree_muzero_recurrent/setup.py build_ext --inplace