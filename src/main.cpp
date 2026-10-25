#include "config.hpp"
#include "matrix.hpp"
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
	try {
		matmul::Config config = matmul::parse_args(argc, argv);
		
		std::cout << "mode=" << matmul::mode_to_string(config.mode)
			  << " n=" << config.matrix_size
			  << " block=" << config.block_size
			  << " reps=" << config.reps
			  << " verify=" << config.verify
			  << " seed=" << config.seed
			  << std::endl;

		if (config.matrix_size <= 8) {
			matmul::Matrix A = matmul::random_matrix(config.matrix_size, config.matrix_size, config.seed);
			matmul::Matrix B = matmul::random_matrix(config.matrix_size, config.matrix_size, config.seed + 1);

			std::cout << "A:\n";
			matmul::print_matrix(A, config.matrix_size, config.matrix_size, std::cout);
			std::cout << "B:\n";
			matmul::print_matrix(B, config.matrix_size, config.matrix_size, std::cout);
		}
	} catch (const std::invalid_argument& error) {
		std::cerr << "Error: " << error.what() << std::endl;
		return 1;
	}

	return 0;
}
