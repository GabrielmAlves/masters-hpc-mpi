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

struct Config {
	Mode mode = Mode::Serial;
	std::size_t matrix_size = 1000;
	std::size_t block_size = 64;
	int reps = 1;
	bool verify = false;
	unsigned seed = 42;
};

Config parse_args(int argc, char** argv); ///< Receive command line arguments and returns a Config object
std::string mode_to_string(Mode mode); ///< Converts Mode to string so it can be written to CSV file

} // namespace matmul
