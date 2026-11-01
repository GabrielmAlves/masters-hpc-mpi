#include "timing.hpp"
#include <iomanip>
#include <ostream>

namespace matmul {
	void print_csv_header(std::ostream& out) {
		out << "mode,kernel,N,block,nprocs,nodes,blas_threads,rep,"
		    << "t_dist,t_comp,t_gather,t_total,t_verif,err_rel,ok\n";
	}

	void print_csv_row(std::ostream& out, const CsvRow& row) {
		const std::streamsize old_precision = out.precision();

		out << row.mode << ',' << row.kernel << ','
		    << row.matrix_size << ',' << row.block_size << ','
		    << row.nprocs << ',' << row.nodes << ',' << row.blas_threads << ','
		    << row.rep << ','
		    << std::setprecision(6)
		    << row.times.dist << ',' << row.times.comp << ',' << row.times.gather << ','
		    << row.times.total() << ',';

		// Without verification the last three columns stay empty (read as NaN by pandas)
		if (row.verified) {
			out << row.times.verif << ','
			    << std::setprecision(9) << row.rel_error << ','
			    << (row.ok ? 1 : 0);
		} else {
			out << ",,";
		}
		out << '\n';

		out.precision(old_precision);
	}
} // namespace matmul
