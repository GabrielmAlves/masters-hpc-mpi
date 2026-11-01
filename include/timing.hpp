#pragma once

#include <cstddef>
#include <iosfwd>
#include <string>

namespace matmul {

/// Times of one repetition, in seconds. In MPI runs each one is the maximum over all ranks
struct Timings {
	double dist = 0.0;	///< Bcast of B + Scatterv of A (zero in serial mode)
	double comp = 0.0;	///< Local multiplication
	double gather = 0.0;	///< Gatherv of C (zero in serial mode)
	double verif = 0.0;	///< reference_multiply + compare (not part of total)

	double total() const { return dist + comp + gather; } ///< Main metric: dist + comp + gather
};

/// One line of the CSV output
struct CsvRow {
	std::string mode;
	std::string kernel;
	std::size_t matrix_size = 0;
	std::size_t block_size = 0;
	int nprocs = 1;
	int nodes = 1;
	int blas_threads = 1;
	int rep = 0;
	Timings times;
	bool verified = false;	///< When false, t_verif, err_rel and ok are left empty
	double rel_error = 0.0;
	bool ok = false;
};

void print_csv_header(std::ostream& out); ///< Writes the CSV column names
void print_csv_row(std::ostream& out, const CsvRow& row); ///< Writes one CSV line

} // namespace matmul
