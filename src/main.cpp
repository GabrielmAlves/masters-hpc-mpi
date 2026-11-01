#include "config.hpp"
#include "matrix.hpp"
#include "kernels.hpp"
#include "verify.hpp"
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>

namespace {
	void print_verification(const std::string& name, const matmul::VerifyResult& result) {
		std::cout << "VERIFICACAO " << std::left << std::setw(8) << name << ": "
			  << (result.isResultOk ? "OK   " : "FALHA")
			  << std::scientific << std::setprecision(2)
			  << "   erro_rel = " << result.relativeError
			  << "   erro_abs_max = " << result.maxAbsoluteError
			  << "   tol = " << result.tolerance
			  << std::defaultfloat << std::endl;
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

		// Reference is expensive (same cost as one multiplication): compute once, compare with all three
		const std::vector<double> C_reference = matmul::reference_multiply(A, B, config.matrix_size);

		print_verification("ijk", matmul::compare(C_ijk, C_reference, matmul::DEFAULT_TOLERANCE));
		print_verification("ikj", matmul::compare(C_ikj, C_reference, matmul::DEFAULT_TOLERANCE));
		print_verification("blocked", matmul::compare(C_blocked, C_reference, matmul::DEFAULT_TOLERANCE));
	} catch (const std::invalid_argument& error) {
		std::cerr << "Error: " << error.what() << std::endl;
		return 1;
	}

	return 0;
}
