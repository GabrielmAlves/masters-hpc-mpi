#pragma once
#include "matrix.hpp"
#include <cstddef>
#include <vector>

namespace matmul {
	constexpr double DEFAULT_TOLERANCE = 1e-4;
	std::vector<double> reference_multiply(const Matrix& A, const Matrix& B, std::size_t n);
struct VerifyResult {
	double relativeError, maxAbsoluteError, tolerance;
	bool isResultOk;
};
	VerifyResult compare(const Matrix& C, const std::vector<double>& C_reference, double tolerance);
} // namespace matmul
