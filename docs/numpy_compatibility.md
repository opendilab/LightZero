# NumPy compatibility

The full LightZero installation currently requires `numpy>=1.24.1,<2`.
Removing this constraint alone does not enable NumPy 2: the released
DI-engine 0.5.3 distribution independently requires `numpy>=1.18.0,<2` and
`gym==0.25.1`. Pip must satisfy both projects' requirements. The constraint
in LightZero also does not override requirements declared by downstream
applications. See [issue #508](https://github.com/opendilab/LightZero/issues/508).

## Native MCTS versus the training stack

The six Cython MuZero-family tree wrappers exchange Python lists and C++
vectors; they do not `cimport numpy`. Although `setup.py` includes NumPy's
headers in its build configuration, that alone does not establish a NumPy
C-ABI dependency.

The `Native MCTS NumPy compatibility` workflow builds those extensions with
NumPy 1.26 and NumPy 2 and tests each build with both runtime versions. It
checks all six binary imports and exercises batched MuZero traversal and
backpropagation with data converted from NumPy arrays. Missing binaries
fail the job, rather than silently skipping tests.

This is a native-extension compatibility gate. It does not certify
DI-engine, Gym environments, replay buffers, neural networks, the separate
pybind11 AlphaZero trees, or end-to-end training under NumPy 2. The tests
load binaries directly to avoid importing the DI-engine-dependent
`lzero.mcts` package initializer.

To reproduce in a disposable Python 3.11 environment:

```bash
python -m pip install 'setuptools<81' wheel 'Cython>=3' pytest pytest-timeout 'numpy>=2,<3'
python setup.py build_ext --inplace
python -m pytest -q tests/test_ctree_numpy_compatibility.py
```

## Completing the migration

A full NumPy 2 migration needs a compatible DI-engine release (including
its dependency constraints), an audit of legacy Gym paths and other
dependencies, and passing environment, replay, model, and training tests.
After those checks pass, LightZero's runtime upper bound can be lifted and
full-stack NumPy 2 CI added. The native-extension matrix provides one part
of that validation; it deliberately leaves the current runtime requirement
in place until the full stack is ready.

Compatibility contributions are welcome. Please include dependency
versions, a minimal reproduction, and separate evidence for native tree
tests and training/evaluation. For applications that already require
NumPy 2, use a separate NumPy 1 environment for the supported LightZero
training stack until the dependency conflict is resolved. Installing with
`--no-deps` bypasses the resolver and is not evidence of supported operation.
