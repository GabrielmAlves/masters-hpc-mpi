#include "config.hpp"
#include "matrix.hpp"
#include "kernels.hpp"
#include <iostream>
#include <stdexcept>
#include <cmath>

namespace {
	float max_abs_diff(const matmul::Matrix& X, const matmul::Matrix& Y) {
		float max_diff = 0.0f;
        	for (std::size_t idx = 0; idx < X.size(); ++idx) {
                	const float diff = std::fabs(X[idx] - Y[idx]);
                	if (diff > max_diff) {
                        	max_diff = diff;
                	}
        	}
        	return max_diff;
	}
}

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

		matmul::Matrix A = matmul::random_matrix(config.matrix_size, config.matrix_size, config.seed);
		matmul::Matrix B = matmul::random_matrix(config.matrix_size, config.matrix_size, config.seed + 1);

		// Kernels accumulate into C, so each result starts zeroed
		matmul::Matrix C_ijk(config.matrix_size*config.matrix_size, 0.0f);
		matmul::Matrix C_ikj(config.matrix_size*config.matrix_size, 0.0f);
		matmul::Matrix C_blocked(config.matrix_size*config.matrix_size, 0.0f);

		matmul::multiply_ijk(A, B, C_ijk, config.matrix_size, config.matrix_size);
		matmul::multiply_ikj(A, B, C_ikj, config.matrix_size, config.matrix_size);
		matmul::multiply_blocked(A, B, C_blocked, config.matrix_size, config.matrix_size, config.block_size);

		if (config.matrix_size <= 8) {
			std::cout << "A:\n";
			matmul::print_matrix(A, config.matrix_size, config.matrix_size, std::cout);
			std::cout << "B:\n";
			matmul::print_matrix(B, config.matrix_size, config.matrix_size, std::cout);
			std::cout << "C (ijk):\n";
			matmul::print_matrix(C_ijk, config.matrix_size, config.matrix_size, std::cout);
		}

		std::cout << "max |C_ijk - C_ikj|     = " << max_abs_diff(C_ijk, C_ikj) << '\n'
			  << "max |C_ijk - C_blocked| = " << max_abs_diff(C_ijk, C_blocked) << std::endl;
	} catch (const std::invalid_argument& error) {
		std::cerr << "Error: " << error.what() << std::endl;
		return 1;
	}

	return 0;
}
