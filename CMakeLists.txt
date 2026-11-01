cmake_minimum_required(VERSION 3.18)
project(masters_hpc_mpi LANGUAGES CXX)
if (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra)
# AVX2 (8 floats per instruction) on both WSL and Lovelace; no FMA contraction, so kernels stay bit-identical
add_compile_options(-march=x86-64-v3 -ffp-contract=off)

find_package(MPI REQUIRED)

find_package(OpenMP REQUIRED)
find_library(OPENBLAS_LIBRARY NAMES openblas REQUIRED)
message(STATUS "OpenBLAS encontrado em: ${OPENBLAS_LIBRARY}")

find_path(CBLAS_INCLUDE_DIR cblas.h REQUIRED)
message(STATUS "cblas.h encontrado em: ${CBLAS_INCLUDE_DIR}")

add_subdirectory(src)
add_subdirectory(tests)
