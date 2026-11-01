#pragma once
#include "matrix.hpp"
#include <cstddef>

namespace matmul {
	void multiply_ijk(const Matrix& A, const Matrix& B, Matrix& C, std::size_t rows, std::size_t n);
	void multiply_ikj(const Matrix& A, const Matrix& B, Matrix& C, std::size_t rows, std::size_t n);
	void multiply_blocked(const Matrix& A, const Matrix& B, Matrix& C, std::size_t rows, std::size_t n, std::size_t block);
	/// Same block loops as multiply_blocked, but each block product is one cblas_sgemm call.
	/// With block >= n (and >= rows) it becomes a single sgemm over the whole strip
	void multiply_blocked_blas(const Matrix& A, const Matrix& B, Matrix& C, std::size_t rows, std::size_t n, std::size_t block);
}
