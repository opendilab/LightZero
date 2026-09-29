#!/usr/bin/env bash
set -euo pipefail

# Build against the active Python interpreter. Override it with PYTHON=/path/to/python.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${script_dir}/build"
python_bin="${PYTHON:-python}"
pybind11_cmake_dir="$(${python_bin} -c 'import pybind11; print(pybind11.get_cmake_dir())')"

cmake -S "${script_dir}" -B "${build_dir}" \
    -DPython3_EXECUTABLE="$(command -v "${python_bin}")" \
    -DCMAKE_PREFIX_PATH="${pybind11_cmake_dir}" \
    "$@"
cmake --build "${build_dir}" --parallel
