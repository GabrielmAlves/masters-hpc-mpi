if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
	message(FATAL_ERROR "Execute cmake na raiz do projeto: cmake -S . -B build")
endif()
add_executable(matmul main.cpp config.cpp matrix.cpp kernels.cpp verify.cpp timing.cpp distribution.cpp)
target_include_directories(matmul PRIVATE ${PROJECT_SOURCE_DIR}/include ${CBLAS_INCLUDE_DIR})
# OpenMP is needed because the Lovelace OpenBLAS is a static library built with OpenMP
target_link_libraries(matmul PRIVATE MPI::MPI_CXX ${OPENBLAS_LIBRARY} OpenMP::OpenMP_CXX)
