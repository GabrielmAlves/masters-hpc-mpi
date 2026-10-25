#include "config.hpp"
#include <stdexcept>
#include <string>
#include <map>

namespace matmul {

namespace {
	const std::map<std::string, Mode> kModes = {
		{"serial", Mode::Serial},
		{"mpi", Mode::Mpi},
		{"mpi_blas", Mode::MpiBlas}
	};

	std::string next_value(int argc, char** argv, int& i, const std::string& option) {
		if (i + 1 >= argc) {
			throw std::invalid_argument("Missing value for " + option);
		}
		++i;
		return argv[i];
	}

	unsigned long to_unsigned(const std::string& text, const std::string& option) {
		bool all_digits = !text.empty();
		
		for (char character : text) {
			if (character < '0' || character > '9') {
				all_digits = false;
			}
		}
		if (!all_digits) {
			throw std::invalid_argument("Invalid value for " + option + ": " + text);
		}

		try {
			return std::stoul(text);
		} catch (const std::out_of_range&) {
			throw std::invalid_argument("Value too large for " + option + ": " + text);
		}
	}
} // namespace
	Config parse_args(int argc, char** argv) {
		Config config;

		for (int i = 1; i < argc; ++i) {
			std::string argument = argv[i];
			
			if (argument == "--verify") {
				config.verify = true;
			} else if (argument == "--mode") {
				std::string value = next_value(argc, argv, i, argument);
				auto iterator = kModes.find(value);
				if (iterator == kModes.end()) {
					throw std::invalid_argument("Invalid mode: " + value + 
								    " (expected serial, mpi or mpi_blas)");
				}
				config.mode = iterator->second;
			} else if (argument == "--n") {
				config.matrix_size = to_unsigned(next_value(argc, argv, i, argument), argument);
			} else if (argument == "--block") {
				config.block_size = to_unsigned(next_value(argc, argv, i, argument), argument);
			} else if (argument == "--reps") {
				config.reps = static_cast<int>(to_unsigned(next_value(argc, argv, i, argument), argument));
			} else if (argument == "--seed") {
				config.seed = static_cast<unsigned>(to_unsigned(next_value(argc, argv, i, argument),argument));
			} else {
				throw std::invalid_argument("Unknown argument: " + argument);
			}
		}

		if (config.matrix_size == 0) {
			throw std::invalid_argument("--n must be greater than zero");
		}
		if (config.block_size == 0) {
			throw std::invalid_argument("--block must be greater than zero");
		}
		if (config.reps <= 0) {
			throw std::invalid_argument("--reps must be greater than zero");
		}
		
		return config;
	}

	std::string mode_to_string(Mode mode) {
		switch (mode) {
			case Mode::Serial: return "serial";
			case Mode::Mpi: return "mpi";
			case Mode::MpiBlas: return "mpi_blas";
		}
		throw std::invalid_argument("Unknown mode..");
	}
} // namespace matmul
