# masters-hpc-mpi

Multiplicação de matrizes densas `C = A × B` (`float`, N×N) em C++17 com MPI e OpenBLAS. Trabalho da disciplina de Computação de Alto Desempenho do mestrado na UNICAMP (Prof. Hervé Yviquel).

Um único executável (`matmul`) oferece três versões:

| Modo | O que faz |
|---|---|
| `--mode serial` | 1 processo, laços próprios: `ijk`, `ikj` ou `blocked` (em blocos b×b), escolhidos com `--kernel` |
| `--mode mpi` | P processos; cada um recebe uma faixa de linhas de A e a matriz B inteira e multiplica com o kernel em blocos |
| `--mode mpi_blas` | Mesma distribuição; a conta local usa `cblas_sgemm` do OpenBLAS, bloco a bloco (com `--block ≥ N`, uma chamada só) |

O relatório está em [`report/relatorio.pdf`](report/relatorio.pdf). As decisões de projeto e o ambiente estão em [`DECISOES.md`](DECISOES.md).

---

## Estrutura

```
CMakeLists.txt          projeto CMake (C++17, Release, -march=x86-64-v3 -ffp-contract=off)
include/  src/          código: config (argumentos), matrix, kernels, distribution, verify, timing, main
tests/                  hello_mpi e hello_blas (testes de fumaça do MPI e do OpenBLAS)
scripts/                scripts PBS do Lovelace + common.sh (módulos, threads, CSV, mpirun)
results/                CSVs (<experimento>_<jobid>.csv) e metadados (.meta) dos jobs do Lovelace
analysis/plots.py       resumos (mediana, mínimo, máximo) e figuras a partir dos CSVs
analysis/figures/       tabelas summary_*.csv e figuras geradas
report/                 relatório (Markdown + PDF) e o script que gera o HTML para impressão
```

---

## Dependências

| Ferramenta | Lovelace (CENAPAD-SP) | Local (WSL2, Ubuntu 24.04) |
|---|---|---|
| Compilador C++17 | GCC 12.2.0 | GCC 13.3.0 |
| MPI | Open MPI 5.0.6 | Open MPI 4.1.6 |
| OpenBLAS | 0.3.21 (estático, OpenMP) | 0.3.26 (`libopenblas-dev`, pthreads) |
| CMake | 3.31.8 | 3.28.3 |
| Python + matplotlib | — | 3.12 (só para a análise) |

No Ubuntu: `sudo apt install build-essential cmake libopenmpi-dev openmpi-bin libopenblas-dev`.

---

## Compilar

### Localmente

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

O executável fica em `build/src/matmul` e os testes de fumaça em `build/tests/`.

### No Lovelace

Os módulos são carregados pelo `scripts/common.sh`. Compile **num nó de cálculo**, pelo PBS, a partir da raiz do projeto:

```bash
qsub scripts/build.pbs
```

O job faz um build limpo e roda três verificações:
- `hello_mpi` com 4 processos;
- `hello_blas` com 4 threads;
- 6 multiplicações pequenas com `--verify`, que devem terminar com `ok = 1`.

Para compilar à mão, por exemplo num job interativo:

```bash
module purge
module load gcc/12.2.0 openmpi/5.0.6-gcc-12.2.0 openblas/0.3.21-gcc-12.2.0 cmake/3.31.8-gcc-12.2.0
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j 4
```

---

## Executar

```bash
mpirun -np <P> ./build/src/matmul --mode <serial|mpi|mpi_blas> --n <N> [opções]
```

| Opção | Padrão | Significado |
|---|---|---|
| `--mode` | `serial` | Versão a executar |
| `--n` | 1000 | Dimensão N das matrizes |
| `--block` | 64 | Tamanho b do bloco (`mpi_blas` com b ≥ N: uma chamada de `sgemm` por processo) |
| `--reps` | 1 | Repetições medidas (uma linha de CSV por repetição) |
| `--verify` | desligado | Compara com uma referência em `double` na 1ª repetição; o tempo é medido à parte (`t_verif`) |
| `--kernel` | `blocked` | Só no modo `serial`: `ijk`, `ikj` ou `blocked` |
| `--blas-threads` | 1 | Só no modo `mpi_blas`: threads do OpenBLAS por processo |
| `--seed` | 42 | Semente de A (B usa `seed + 1`) |
| `--header` | desligado | Imprime o cabeçalho do CSV antes das linhas |

Exemplos:

```bash
mpirun -np 1 ./build/src/matmul --mode serial --kernel ikj --n 1000 --verify --header
mpirun -np 4 ./build/src/matmul --mode mpi --n 2000 --block 128 --reps 3 --header
mpirun -np 4 ./build/src/matmul --mode mpi_blas --n 2000 --block 2000 --verify --header
```

A saída é CSV, no `stdout`:

```
mode,kernel,N,block,nprocs,nodes,blas_threads,rep,t_dist,t_comp,t_gather,t_total,t_verif,err_rel,ok
```

- Os tempos medidos são três:
  - `t_dist`: `MPI_Bcast` de B + `MPI_Scatterv` de A;
  - `t_comp`: multiplicação local;
  - `t_gather`: `MPI_Gatherv` de C.

  Cada um é o **máximo entre os processos**, e `t_total` é a soma dos três.
- A geração das matrizes e a verificação **não** entram em `t_total`.
- `err_rel` é o erro relativo na norma de Frobenius; `ok = 1` se `err_rel < 1e-4`.

### Threads do OpenBLAS

O número de threads é fixado de forma explícita em três lugares:
- `OMP_NUM_THREADS` no `scripts/common.sh` (padrão 1);
- `OPENBLAS_NUM_THREADS` no mesmo arquivo (padrão 1);
- `openblas_set_num_threads(--blas-threads)` no programa.

O valor efetivo, lido com `openblas_get_num_threads()`, é gravado na coluna `blas_threads`. No Lovelace o OpenBLAS usa OpenMP e obedece ao `OMP_NUM_THREADS`. No WSL ele usa pthreads e obedece ao `OPENBLAS_NUM_THREADS`.

---

## Experimentos no Lovelace

Submeta da raiz do projeto, depois do `build.pbs`. Cada job grava `results/<experimento>_<jobid>.csv` e um `.meta` com nó, data, commit, módulos e threads.

| Script | Fila | O que varia |
|---|---|---|
| `pilot.pbs` | `testes` | 1 repetição das configurações mais lentas (para dimensionar walltimes) |
| `correctness.pbs` | `testes` | N ∈ {2000, 4000, 6000}, P ∈ {1, 3, 4}, todas as versões com `--verify` |
| `serial.pbs` | `par128` (rodado com `-q serial`) | kernels `ijk`, `ikj`, `blocked` (b ∈ {32, 64, 128}), N ∈ {500, 1000, 2000} |
| `blocks.pbs` | `par128` | b ∈ {32, …, 512, N}, P ∈ {1, 16, 128}, N = 6000 |
| `strong.pbs` | `par128` | P = 1, 2, 4, …, 128 com N ∈ {2000, 6000} fixo |
| `sizes.pbs` | `par128` | N ∈ {2000, 4000, 6000}, P ∈ {1, 16, 128} |
| `weak.pbs` | `par128` | P de 1 a 128 com N = 2000·∛P (trabalho por processo constante) |
| `hybrid.pbs` | `par128` | P × T ∈ {128×1, 64×2, 32×4, 16×8, 8×16} (processos × threads do OpenBLAS) |

```bash
qsub scripts/correctness.pbs
qsub scripts/strong.pbs
```

As listas podem ser trocadas sem editar os scripts, com `qsub -v`. As opções `-q` e `-l` substituem as linhas `#PBS`:

```bash
qsub -v NS="6000",PS="1 2 4" scripts/strong.pbs
qsub -q testes -l select=1:ncpus=4:mpiprocs=4 -l walltime=00:30:00 -v PAIRS="1:2000 2:2520 4:3175" scripts/weak.pbs
```

Para rodar os scripts fora do cluster: `LOCAL_TEST=1 MPIRUN_EXTRA=--oversubscribe bash scripts/correctness.pbs`.

---

## Análise

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install matplotlib
python3 analysis/plots.py --results results --out analysis/figures
```

O script lê todos os `<experimento>_<jobid>.csv`, agrega as repetições (mediana, mínimo e máximo) e gera `summary_<experimento>.csv` e as figuras.

As figuras do relatório usam só os jobs da fila `par128`. Os jobs 1035736 e 1035737, versões reduzidas rodadas na fila `testes`, ficaram de fora:

```bash
mkdir -p /tmp/res_main
for f in results/*_[0-9]*.csv; do
  case "$f" in *1035736*|*1035737*) ;; *) ln -sf "$PWD/$f" /tmp/res_main/ ;; esac
done
python3 analysis/plots.py --results /tmp/res_main --out analysis/figures
```
