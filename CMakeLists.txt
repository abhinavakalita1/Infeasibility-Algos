cmake_minimum_required(VERSION 3.15)
project(manifold_tracing)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Find dependencies ────────────────────────────────────────────────────────
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)      # pip install pybind11 / conda install pybind11
find_package(Eigen3 REQUIRED)        # apt install libeigen3-dev / conda install eigen

# ── Extension module ─────────────────────────────────────────────────────────
pybind11_add_module(manifold_tracing
    manifold_tracing_bind.cpp
)

target_include_directories(manifold_tracing PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}      # picks up utils.h, permutahedral_simplex.h, etc.
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(manifold_tracing PRIVATE
    Eigen3::Eigen
)

# Optional: faster, smaller release builds
target_compile_options(manifold_tracing PRIVATE -O3 -march=native)