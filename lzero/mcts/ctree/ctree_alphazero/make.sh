#!/bin/bash

# This script compiles the ctree_alphazero project.
# The compiled files are stored in the "build" directory.
#
# In summary, this script automates the process of creating a new build directory,
# navigating into it, running cmake to generate build files suitable for the arm64 architecture,
# and running make to compile the project.

# Navigate to the project directory.
# ========= NOTE: PLEASE MODIFY THE FOLLOWING DIRECTORY TO YOUR OWN. =========
cd /YOUR_LightZero_DIR/LightZero/lzero/mcts/ctree/ctree_alphazero/ || exit

# Create a new directory named "build." The build directory is where the compiled files will be stored.
mkdir -p build

# Navigate into the "build" directory
cd build || exit

# Run cmake on the parent directory with the specified architecture
cmake .. -DCMAKE_OSX_ARCHITECTURES="arm64"

# Run the "make" command to compile the project
make