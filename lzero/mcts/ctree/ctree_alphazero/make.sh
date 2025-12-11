#!/bin/bash

# This script compiles the ctree_alphazero project.
# The compiled files are stored in the "build" directory.
#
# In summary, this script automates the process of creating a new build directory,
# navigating into it, running cmake to generate build files suitable for the arm64 architecture,
# and running make to compile the project.

# Function to find the ctree_alphazero directory
find_ctree_alphazero_dir() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    # Check if we're already in the ctree_alphazero directory
    if [[ "$script_dir" == */lzero/mcts/ctree/ctree_alphazero ]]; then
        echo "$script_dir"
        return 0
    fi

    # Try to find the directory by searching upwards from script location
    local current_dir="$script_dir"
    while [[ "$current_dir" != "/" ]]; do
        if [[ -d "$current_dir/lzero/mcts/ctree/ctree_alphazero" ]]; then
            echo "$current_dir/lzero/mcts/ctree/ctree_alphazero"
            return 0
        fi
        current_dir="$(dirname "$current_dir")"
    done

    # Try to find from current working directory
    if [[ -d "./lzero/mcts/ctree/ctree_alphazero" ]]; then
        echo "$(pwd)/lzero/mcts/ctree/ctree_alphazero"
        return 0
    fi

    # Check if CMakeLists.txt exists in current directory (maybe we're already there)
    if [[ -f "./CMakeLists.txt" ]] && [[ -f "./alphazero_mcts_cpp.cpp" ]]; then
        echo "$(pwd)"
        return 0
    fi

    return 1
}

# Navigate to the project directory.
CTREE_DIR=$(find_ctree_alphazero_dir)

if [[ -z "$CTREE_DIR" ]]; then
    echo "Error: Could not find the ctree_alphazero directory."
    echo "Please ensure you are running this script from within the LightZero project,"
    echo "or manually specify the correct path in the script."
    echo ""
    echo "Expected directory structure: LightZero/lzero/mcts/ctree/ctree_alphazero/"
    exit 1
fi

echo "Found ctree_alphazero directory: $CTREE_DIR"
cd "$CTREE_DIR" || exit

# Create a new directory named "build." The build directory is where the compiled files will be stored.
mkdir -p build

# Navigate into the "build" directory
cd build || exit

# Get pybind11 cmake directory
PYBIND11_CMAKE_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")

# Run cmake on the parent directory with the specified architecture and pybind11 path
cmake .. -DCMAKE_OSX_ARCHITECTURES="arm64" -DCMAKE_PREFIX_PATH="$PYBIND11_CMAKE_DIR"

# Run the "make" command to compile the project
make