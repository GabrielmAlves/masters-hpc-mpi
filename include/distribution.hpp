#pragma once

#include <cstddef>
#include <vector>

namespace matmul {

/// How the rows of an N x N matrix are split among P processes (layout used by Scatterv/Gatherv)
struct RowDistribution {
	std::vector<int> rows;		///< rows[r]: number of rows of rank r
	std::vector<int> counts;	///< counts[r] = rows[r] * N, in floats
	std::vector<int> displs;	///< displs[r]: offset of the first float of rank r
};

/// Splits n rows among nprocs ranks: rows[r] = n/P + (r < n%P ? 1 : 0).
/// Throws std::invalid_argument if n*n does not fit in an int (MPI counts are int)
RowDistribution distribute_rows(std::size_t n, int nprocs);

} // namespace matmul
