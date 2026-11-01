#pragma once

#include <string>
#include <cstddef>

namespace matmul {


/// Enum class representing which mode of the multiplication will be performed
enum class Mode {
	Serial,
	Mpi,
	MpiBlas
};

/// Enum class representing which serial kernel is used (only with --mode serial)
enum class Kernel {
	Ijk,
	Ikj,
	Blocked
};

struct Config {
	Mode mode = Mode::Serial;
	std::size_t matrix_size = 1000;
	std::size_t block_size = 64;
	int reps = 1;
	bool verify = false;
	bool header = false;
	unsigned seed = 42;
	Kernel kernel = Kernel::Blocked;
	int blas_threads = 1;	///< --blas-threads: OpenBLAS threads per process (only with --mode mpi_blas)
};

Config parse_args(int argc, char** argv); ///< Receive command line arguments and returns a Config object
std::string mode_to_string(Mode mode); ///< Converts Mode to string so it can be written to CSV file
std::string kernel_to_string(Kernel kernel); ///< Converts Kernel to string so it can be written to CSV file
} // namespace matmul
