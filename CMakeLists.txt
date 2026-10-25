cmake_minimum_required(VERSION 3.18)
project(masters_hpc_mpi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

find_package(OpenMP REQUIRED)
find_library(OPENBLAS_LIBRARY NAMES openblas REQUIRED)
message(STATUS "OpenBLAS encontrado em: $OPENBLAS_LIBRARY")

find_path(CBLAS_INCLUDE_DIR cblas.h REQUIRED)
message(STATUS "cblas.h encontrado em: ${CBLAS_INCLUDE_DIR}")

add_subdirectory(tests)
