#include "verify.hpp"
#include <cmath>
#include <algorithm>

namespace matmul {
	std::vector<double> reference_multiply(const Matrix& A, const Matrix& B, std::size_t n) {
		std::vector<double> C_reference(n*n, 0.0);

		for (std::size_t i = 0; i < n; ++i) {
			for (std::size_t k = 0; k < n; ++k) {
				const double a = A[i*n + k];
				for (std::size_t j = 0; j < n; ++j) {
					C_reference[i*n + j] += a*B[k*n + j];
				}
			}
		}

		return C_reference;
	}

	VerifyResult compare(const Matrix& C, const std::vector<double>& C_reference, double tolerance) {
		double diff_sq = 0.0;
		double reference_sq = 0.0;
		double max_abs = 0.0;

		for (std::size_t index = 0; index < C_reference.size(); ++index) {
			const double difference = static_cast<double>(C[index]) - C_reference[index];
			diff_sq += difference * difference;
			reference_sq += C_reference[index]*C_reference[index];
			max_abs = std::max(max_abs, std::fabs(difference));
		}

		const double relative_error = std::sqrt(diff_sq) / std::sqrt(reference_sq);

		return {relative_error, max_abs, tolerance, relative_error < tolerance};
	}
} // namespace matmul
