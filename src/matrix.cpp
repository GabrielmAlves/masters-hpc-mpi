#include "matrix.hpp"
#include <iomanip>
#include <random>
#include <iostream>

namespace matmul {
	Matrix random_matrix(std::size_t rows, std::size_t columns, unsigned seed) {
		Matrix matrix(rows*columns);

		std::mt19937 generator(seed);
		std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

		for (float& value : matrix) {
			value = distribution(generator);
		}

		return matrix;
	}

	void print_matrix(const Matrix& matrix, std::size_t rows, std::size_t columns, std::ostream& out) {
		for (std::size_t i = 0; i < rows; ++i) {
			for (std::size_t j = 0; j < columns; ++j) {
				out << std::setw(10) << matrix[i*columns + j] << ' ';
			}
			out << '\n';
		}
	}
}
