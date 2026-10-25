#include "matrix.hpp"
#include <cstddef>

namespace matmul {
	void multiply_ijk(const Matrix& A, const Matrix& B, Matrix& C, std::size_t rows, std::size_t n);
	void multiply_ikj(const Matrix& A, const Matrix& B, Matrix& C, std::size_t rows, std::size_t n);
	void multiply_blocked(const Matrix& A, const Matrix& B, Matrix& C, std::size_t rows, std::size_t n, std::size_t block);
}
