#include "config.hpp"
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
	} catch (const std::invalid_argument& error) {
		std::cerr << "Error: " << error.what() << std::endl;
		return 1;
	}

	return 0;
}
