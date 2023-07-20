# cmake_minimum_required(VERSION 2.8.12)
cmake_minimum_required(VERSION 3.4...3.18)

# cmake_policy(SET CMP0048 NEW)
project(pybind11_example VERSION 1.0)

# 找到 Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

add_subdirectory(pybind11)
pybind11_add_module(pybind11_example pybind11_example.cpp)

# 将Python头文件路径添加到你的库的包含路径中
target_include_directories(pybind11_example PRIVATE ${Python3_INCLUDE_DIRS})

# 链接你的库和 pybind11::module
target_link_libraries(pybind11_example PRIVATE pybind11::module)

# 设置你的库的Python版本
set_target_properties(pybind11_example PROPERTIES PYTHON_STANDARD ${Python3_VERSION})



# CMakeLists.txt