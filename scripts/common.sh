# Shared setup for the PBS scripts: modules, BLAS threads, CSV output and the run helper.
# Sourced by every scripts/*.pbs from the project root ($PBS_O_WORKDIR).

MODULES="gcc/12.2.0 openmpi/5.0.6-gcc-12.2.0 openblas/0.3.21-gcc-12.2.0 cmake/3.31.8-gcc-12.2.0"

if command -v module >/dev/null 2>&1; then
	module purge
	module load $MODULES
elif [ "${LOCAL_TEST:-0}" = "1" ]; then
	echo "LOCAL_TEST=1: skipping module load" >&2
else
	echo "Error: 'module' command not found (set LOCAL_TEST=1 to run outside Lovelace)" >&2
	exit 1
fi

# One BLAS thread per MPI process. Both variables, because the pthreads OpenBLAS (WSL) reads
# OPENBLAS_NUM_THREADS and the OpenMP one (Lovelace) reads OMP_NUM_THREADS.
# The program also fixes it in the code (--blas-threads, default 1)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

BIN=./build/src/matmul
JOB_ID=${PBS_JOBID:-local}
JOB_ID=${JOB_ID%%.*}

# Extra mpirun options, empty on Lovelace (locally: MPIRUN_EXTRA=--oversubscribe)
MPIRUN_EXTRA=${MPIRUN_EXTRA:-}

# start_csv NAME: creates results/NAME_<jobid>.csv with the header, plus a .meta file describing the run
start_csv() {
	if [ ! -x "$BIN" ]; then
		echo "Error: $BIN not found; submit scripts/build.pbs first" >&2
		exit 1
	fi
	mkdir -p results
	OUT="results/${1}_${JOB_ID}.csv"
	mpirun -np 1 $MPIRUN_EXTRA "$BIN" --header --n 1 | head -n 1 > "$OUT"
	{
		echo "job: ${PBS_JOBID:-local}"
		echo "script: $1"
		echo "date: $(date -Iseconds)"
		echo "host: $(hostname)"
		echo "commit: $(git rev-parse --short HEAD 2>/dev/null)$(git diff --quiet HEAD 2>/dev/null || echo ' (uncommitted changes)')"
		echo "modules: $MODULES"
		echo "OMP_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
	} > "${OUT%.csv}.meta"
	echo "Writing $OUT"
}

# run P ARGS...: one matmul execution with P processes, one per core, appended to $OUT.
# A failed execution is reported in the job log and the script goes on with the next one
run() {
	local procs=$1
	shift
	echo "[$(date +%T)] P=$procs $*"
	if ! mpirun -np "$procs" --bind-to core --map-by core $MPIRUN_EXTRA \
		-x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS \
		"$BIN" "$@" >> "$OUT"; then
		echo "FAILED: P=$procs $*" >&2
	fi
}
