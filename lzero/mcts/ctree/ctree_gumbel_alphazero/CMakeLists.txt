# Declare the minimum version of CMake that can be used
# To understand and build the project
cmake_minimum_required(VERSION 3.4...3.18)

# Set the project name to mcts_gumbel_alphazero and set the version to 1.0
project(mcts_gumbel_alphazero VERSION 1.0)

# Find and get the details of Python package
# This is required for embedding Python in the project
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Add pybind11 as a subdirectory,
# so that its build files are generated alongside the current project.
# This is necessary because the current project depends on pybind11
# Specify the source directory and build directory of pybind11
add_subdirectory(
    /Users/<your_user_name>/code/LightZero/lzero/mcts/ctree/ctree_alphazero/pybind11 # TODO: Change this to the path of your pybind11 source directory
    ${CMAKE_CURRENT_BINARY_DIR}/pybind11_build
)
# Add the .cpp file to the mcts_gumbel_alphazero module
# These files are compiled and linked into the module
pybind11_add_module(mcts_gumbel_alphazero mcts_gumbel_alphazero.cpp)

# Add the Python header file paths to the include paths
# of the mcts_gumbel_alphazero library. This is necessary for the
# project to find the Python header files it needs to include
target_include_directories(mcts_gumbel_alphazero PRIVATE ${Python3_INCLUDE_DIRS})

# Link the mcts_gumbel_alphazero library with the pybind11::module target.
# This is necessary for the mcts_gumbel_alphazero library to use the functions and classes defined by pybind11
target_link_libraries(mcts_gumbel_alphazero PRIVATE pybind11::module)

# Set the Python standard to the version of Python found by find_package(Python3)
# This ensures that the code will be compiled against the correct version of Python
set_target_properties(mcts_gumbel_alphazero PROPERTIES PYTHON_STANDARD ${Python3_VERSION})