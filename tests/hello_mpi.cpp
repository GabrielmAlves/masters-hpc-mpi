#include <mpi.h>
#include <iostream>

int main(int argc, char* argv[]){
	int rank, size, processor_size;
	char name[MPI_MAX_PROCESSOR_NAME];
	
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Get_processor_name(name, &processor_size);
	
	std::cout << "Rank " << rank << " de " << size << " rodando em " << name << std::endl;

	MPI_Finalize();
	
	return 0;
}
