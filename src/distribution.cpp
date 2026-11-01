#include "distribution.hpp"
#include <climits>
#include <stdexcept>
#include <string>

namespace matmul {
	RowDistribution distribute_rows(std::size_t n, int nprocs) {
		if (nprocs <= 0) {
			throw std::invalid_argument("Number of processes must be greater than zero");
		}
		if (n > 0 && n > static_cast<std::size_t>(INT_MAX) / n) {
			throw std::invalid_argument("--n too large: N*N = " + std::to_string(n) + "^2 does not fit in an MPI count (int)");
		}

		const std::size_t procs = static_cast<std::size_t>(nprocs);
		const std::size_t base = n / procs;
		const std::size_t extra = n % procs;

		RowDistribution distribution;
		distribution.rows.resize(procs);
		distribution.counts.resize(procs);
		distribution.displs.resize(procs);

		std::size_t offset = 0;
		for (std::size_t r = 0; r < procs; ++r) {
			const std::size_t rows = base + (r < extra ? 1 : 0);
			distribution.rows[r] = static_cast<int>(rows);
			distribution.counts[r] = static_cast<int>(rows * n);
			distribution.displs[r] = static_cast<int>(offset);
			offset += rows * n;
		}

		return distribution;
	}
} // namespace matmul
