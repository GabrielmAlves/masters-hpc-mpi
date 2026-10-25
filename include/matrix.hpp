#pragma once

#include <vector>
#include <cstddef>
#include <iosfwd>

namespace matmul {
	using Matrix = std::vector<float>;
	
	Matrix random_matrix(std::size_t rows, std::size_t columns, unsigned seed);
	void print_matrix(const Matrix& matrix, std::size_t rows, std::size_t columns, std::ostream& out);
}
