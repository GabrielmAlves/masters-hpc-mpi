add_executable(matmul main.cpp config.cpp matrix.cpp kernels.cpp)
target_include_directories(matmul PRIVATE ${PROJECT_SOURCE_DIR}/include)
