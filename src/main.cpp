#include "config.hpp"
#include "matrix.hpp"
#include "kernels.hpp"
#include "verify.hpp"
#include "timing.hpp"
#include "distribution.hpp"
#include <mpi.h>
#include <cblas.h>
#include <algorithm>
#include <exception>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
	void run_serial_kernel(const matmul::Config& config, const matmul::Matrix& A, const matmul::Matrix& B, matmul::Matrix& C) {
		const std::size_t n = config.matrix_size;
		switch (config.kernel) {
			case matmul::Kernel::Ijk: matmul::multiply_ijk(A, B, C, n, n); break;
			case matmul::Kernel::Ikj: matmul::multiply_ikj(A, B, C, n, n); break;
			case matmul::Kernel::Blocked: matmul::multiply_blocked(A, B, C, n, n, config.block_size); break;
		}
	}

	std::string kernel_label(const matmul::Config& config) {
		switch (config.mode) {
			case matmul::Mode::Serial: return matmul::kernel_to_string(config.kernel);
			case matmul::Mode::Mpi: return "blocked";
			case matmul::Mode::MpiBlas: return "sgemm";
		}
		return "unknown";
	}

	int count_nodes(int rank, int size) {
		char name[MPI_MAX_PROCESSOR_NAME] = {};
		int length = 0;
		MPI_Get_processor_name(name, &length);

		std::vector<char> all_names(rank == 0 ? static_cast<std::size_t>(size) * MPI_MAX_PROCESSOR_NAME : 0);
		MPI_Gather(name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
			   all_names.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);

		std::set<std::string> distinct;
		if (rank == 0) {
			for (int r = 0; r < size; ++r) {
				distinct.insert(std::string(&all_names[static_cast<std::size_t>(r) * MPI_MAX_PROCESSOR_NAME]));
			}
		}
		return static_cast<int>(distinct.size());
	}

	matmul::Timings reduce_max(const matmul::Timings& local) {
		double local_values[3] = {local.dist, local.comp, local.gather};
		double max_values[3] = {0.0, 0.0, 0.0};
		MPI_Reduce(local_values, max_values, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

		matmul::Timings result;
		result.dist = max_values[0];
		result.comp = max_values[1];
		result.gather = max_values[2];
		return result;
	}

	matmul::CsvRow make_row(const matmul::Config& config, int size, int nodes) {
		matmul::CsvRow row;
		row.mode = matmul::mode_to_string(config.mode);
		row.kernel = kernel_label(config);
		row.matrix_size = config.matrix_size;
		row.block_size = config.block_size;
		row.nprocs = size;
		row.nodes = nodes;
		row.blas_threads = config.mode == matmul::Mode::MpiBlas ? openblas_get_num_threads() : 1;
		return row;
	}

	void verify_into(matmul::CsvRow& row, const matmul::Matrix& A, const matmul::Matrix& B, const matmul::Matrix& C, std::size_t n) {
		const double tv = MPI_Wtime();
		const std::vector<double> C_reference = matmul::reference_multiply(A, B, n);
		const matmul::VerifyResult result = matmul::compare(C, C_reference, matmul::DEFAULT_TOLERANCE);
		row.times.verif = MPI_Wtime() - tv;
		row.rel_error = result.relativeError;
		row.ok = result.isResultOk;
	}

	void run_serial(const matmul::Config& config, int rank, int size, int nodes) {
		const std::size_t n = config.matrix_size;
		const matmul::Matrix A = matmul::random_matrix(n, n, config.seed);
		const matmul::Matrix B = matmul::random_matrix(n, n, config.seed + 1);
		matmul::Matrix C(n*n);

		matmul::CsvRow row = make_row(config, size, nodes);

		for (int rep = 0; rep < config.reps; ++rep) {
			// Kernels accumulate into C, so it must be zeroed before every repetition
			std::fill(C.begin(), C.end(), 0.0f);

			matmul::Timings local;
			MPI_Barrier(MPI_COMM_WORLD);
			const double t0 = MPI_Wtime();
			run_serial_kernel(config, A, B, C);
			local.comp = MPI_Wtime() - t0;

			row.rep = rep;
			row.times = reduce_max(local);
			row.verified = config.verify && rep == 0;

			if (row.verified) {
				verify_into(row, A, B, C, n);
			}
			if (rank == 0) {
				matmul::print_csv_row(std::cout, row);
			}
		}
	}

	void run_mpi(const matmul::Config& config, int rank, int size, int nodes) {
		const std::size_t n = config.matrix_size;
		const matmul::RowDistribution distribution = matmul::distribute_rows(n, size);
		const std::size_t local_rows = static_cast<std::size_t>(distribution.rows[rank]);
		const int local_count = distribution.counts[rank];

		matmul::Matrix A;
		matmul::Matrix C;
		matmul::Matrix B(n*n);
		if (rank == 0) {
			A = matmul::random_matrix(n, n, config.seed);
			B = matmul::random_matrix(n, n, config.seed + 1);
			C.resize(n*n);
		}
		matmul::Matrix A_local(local_rows*n);
		matmul::Matrix C_local(local_rows*n);

		matmul::CsvRow row = make_row(config, size, nodes);

		for (int rep = 0; rep < config.reps; ++rep) {
			std::fill(C_local.begin(), C_local.end(), 0.0f);

			matmul::Timings local;
			MPI_Barrier(MPI_COMM_WORLD);
			const double t0 = MPI_Wtime();
			MPI_Bcast(B.data(), static_cast<int>(n*n), MPI_FLOAT, 0, MPI_COMM_WORLD);
			MPI_Scatterv(A.data(), distribution.counts.data(), distribution.displs.data(), MPI_FLOAT,
				     A_local.data(), local_count, MPI_FLOAT, 0, MPI_COMM_WORLD);
			const double t1 = MPI_Wtime();
			if (config.mode == matmul::Mode::MpiBlas) {
				matmul::multiply_blocked_blas(A_local, B, C_local, local_rows, n, config.block_size);
			} else {
				matmul::multiply_blocked(A_local, B, C_local, local_rows, n, config.block_size);
			}
			const double t2 = MPI_Wtime();
			MPI_Gatherv(C_local.data(), local_count, MPI_FLOAT,
				    C.data(), distribution.counts.data(), distribution.displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
			const double t3 = MPI_Wtime();

			local.dist = t1 - t0;
			local.comp = t2 - t1;
			local.gather = t3 - t2;

			row.rep = rep;
			row.times = reduce_max(local);
			row.verified = config.verify && rep == 0;

			if (rank == 0) {
				if (row.verified) {
					verify_into(row, A, B, C, n);
				}
				matmul::print_csv_row(std::cout, row);
			}
		}
	}
}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);

	int rank = 0;
	int size = 1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int status = 0;
	try {
		// Every rank parses the same arguments, so an invalid one makes all ranks stop together
		const matmul::Config config = matmul::parse_args(argc, argv);

		if (config.mode == matmul::Mode::Serial && size > 1) {
			throw std::invalid_argument("--mode serial runs with a single process (got " +
						    std::to_string(size) + "); use mpirun -np 1");
		}
		if (config.mode == matmul::Mode::MpiBlas) {
			openblas_set_num_threads(config.blas_threads);
		}

		const int nodes = count_nodes(rank, size);

		if (config.header && rank == 0) {
			matmul::print_csv_header(std::cout);
		}

		if (config.mode == matmul::Mode::Serial) {
			run_serial(config, rank, size, nodes);
		} else {
			run_mpi(config, rank, size, nodes);
		}
	} catch (const std::invalid_argument& error) {
		if (rank == 0) {
			std::cerr << "Error: " << error.what() << std::endl;
		}
		status = 1;
	} catch (const std::exception& error) {
		std::cerr << "Fatal error on rank " << rank << ": " << error.what() << std::endl;
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	MPI_Finalize();
	return status;
}
