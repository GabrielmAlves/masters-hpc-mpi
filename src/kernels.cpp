#include "kernels.hpp"
#include <algorithm>

namespace matmul {
	void multiply_ijk(const Matrix& A, const Matrix& B, Matrix& C, std::size_t rows, std::size_t n) {
		for (std::size_t i = 0; i < rows; ++i) {
			for (std::size_t j = 0; j < n; ++j) {
				float sum = 0.0f;
				for (std::size_t k = 0; k < n; ++k) {
					sum += A[i*n + k]*B[k*n + j];
				}
				C[i*n + j] += sum;
			}
		}
	}

	void multiply_ikj(const Matrix& A, const Matrix& B, Matrix& C, std::size_t rows, std::size_t n) {
		for (std::size_t i = 0; i < rows; ++i) {
                	for (std::size_t k = 0; k < n; ++k) {
                        	const float a = A[i * n + k];
                        	for (std::size_t j = 0; j < n; ++j) {
                                	C[i * n + j] += a * B[k * n + j];
                        	}
                	}
        	}
	}

	void multiply_blocked(const Matrix& A, const Matrix& B, Matrix& C, std::size_t rows, std::size_t n, std::size_t block) {
		for (std::size_t ii = 0; ii < rows; ii += block) {
                	const std::size_t i_end = std::min(ii + block, rows);
                	for (std::size_t kk = 0; kk < n; kk += block) {
                        	const std::size_t k_end = std::min(kk + block, n);
                        	for (std::size_t jj = 0; jj < n; jj += block) {
                                	const std::size_t j_end = std::min(jj + block, n);

                                	for (std::size_t i = ii; i < i_end; ++i) {
                                        	for (std::size_t k = kk; k < k_end; ++k) {
                                                	const float a = A[i * n + k];
                                                	for (std::size_t j = jj; j < j_end; ++j) {
                                                        	C[i * n + j] += a * B[k * n + j];
                                                	}
                                        	}
                                	}
                        	}
                	}
        	}
	}
}
