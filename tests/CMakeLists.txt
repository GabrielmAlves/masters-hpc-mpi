add_executable(hello_mpi hello_mpi.cpp)
target_link_libraries(hello_mpi PRIVATE MPI::MPI_CXX)
