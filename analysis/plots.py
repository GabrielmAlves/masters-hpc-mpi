#!/usr/bin/env python3
"""Summary tables and figures for the matmul experiments.

Reads every <experiment>_<jobid>.csv in the results directory (the Lovelace runs; files
without a job id, such as wsl_*.csv, are ignored), aggregates the repetitions of each
configuration (median, min, max) and writes, for each experiment found, a summary table
(summary_<experiment>.csv) and its figures. Experiments without data are skipped.

Usage:
    python analysis/plots.py [--results results] [--out analysis/figures]

Only needs matplotlib (the CSVs are read with the standard library).
"""

import argparse
import csv
import glob
import math
import os
import re
import statistics
from collections import defaultdict

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots of the reference palette (light mode), assigned in fixed order.
# An entity keeps its color in every figure
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#8a8984"

FILE_PATTERN = re.compile(r"^(?P<experiment>[a-z_]+?)_(?P<job>\d+)\.csv$")
NUMERIC = ["N", "block", "nprocs", "nodes", "blas_threads", "rep",
           "t_dist", "t_comp", "t_gather", "t_total", "t_verif", "err_rel", "ok"]
CONFIG_KEYS = ["mode", "kernel", "N", "block", "nprocs", "blas_threads"]


# ----------------------------------------------------------------------------- data

def load(results_dir):
    """Returns {experiment: [row, ...]} with numeric fields converted (empty -> None)."""
    experiments = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(results_dir, "*.csv"))):
        match = FILE_PATTERN.match(os.path.basename(path))
        if not match:
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                for key in NUMERIC:
                    value = row.get(key, "")
                    row[key] = None if value == "" else float(value)
                for key in ("N", "block", "nprocs", "nodes", "blas_threads", "rep"):
                    row[key] = int(row[key])
                experiments[match["experiment"]].append(row)
    return experiments


def aggregate(rows):
    """One entry per configuration: median/min/max of each time over the repetitions."""
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in CONFIG_KEYS)].append(row)

    summary = []
    for key, reps in groups.items():
        entry = dict(zip(CONFIG_KEYS, key))
        entry["reps"] = len(reps)
        for t in ("t_dist", "t_comp", "t_gather", "t_total"):
            values = [r[t] for r in reps]
            entry[t] = statistics.median(values)
            entry[t + "_min"] = min(values)
            entry[t + "_max"] = max(values)
        verified = [r for r in reps if r["err_rel"] is not None]
        entry["t_verif"] = verified[0]["t_verif"] if verified else None
        entry["err_rel"] = verified[0]["err_rel"] if verified else None
        entry["ok"] = int(verified[0]["ok"]) if verified else None
        entry["gflops"] = 2 * entry["N"] ** 3 / entry["t_total"] / 1e9
        summary.append(entry)
    summary.sort(key=lambda e: (e["mode"], e["kernel"], e["N"], e["block"], e["nprocs"], e["blas_threads"]))
    return summary


def series_name(entry):
    """Legend name of the line/bar an entry belongs to."""
    if entry["mode"] == "serial":
        if entry["kernel"] == "blocked":
            return f"serial blocked (b={entry['block']})"
        return f"serial {entry['kernel']}"
    if entry["mode"] == "mpi":
        return f"mpi (b={entry['block']})"
    if entry["block"] >= entry["N"]:
        return "mpi_blas (sgemm único)"
    return f"mpi_blas (b={entry['block']})"


class Colors:
    """Fixed entity -> color map, filled in order of first appearance and shared by all figures."""

    PRESET = {"mpi (b=64)": SLOTS[0], "mpi (b=128)": SLOTS[1], "mpi_blas (sgemm único)": SLOTS[2],
              "mpi": SLOTS[0], "mpi_blas": SLOTS[2]}

    def __init__(self):
        self.assigned = dict(self.PRESET)

    def __call__(self, name):
        if name not in self.assigned:
            used = set(self.assigned.values())
            free = [c for c in SLOTS if c not in used] or SLOTS
            self.assigned[name] = free[0]
        return self.assigned[name]


COLORS = Colors()


def write_table(entries, path, columns):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for e in entries:
            writer.writerow(["" if e.get(c) is None else (f"{e[c]:.6g}" if isinstance(e[c], float) else e[c])
                             for c in columns])


# ----------------------------------------------------------------------------- style

def setup_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": TEXT, "axes.labelcolor": TEXT, "axes.titlecolor": TEXT,
        "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
        "axes.edgecolor": TEXT_SECONDARY, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "lines.linewidth": 2, "lines.markersize": 6,
        "legend.frameon": False, "font.size": 10, "axes.titlesize": 11,
    })


def finish(fig, path, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  figure: {path}")


def log_ticks(ax, values):
    """Ticks at 1, 2 and 5 x 10^k covering the data, so a log axis always has readable labels."""
    low, high = min(values), max(values)
    ticks = [m * 10.0 ** k for k in range(math.floor(math.log10(low)) - 1, math.ceil(math.log10(high)) + 1)
             for m in (1, 2, 5) if low / 1.5 <= m * 10.0 ** k <= high * 1.5]
    ax.set_yticks(ticks)
    labels = []
    for t in ticks:
        exponent = math.floor(math.log10(t))
        labels.append(r"$%.0f\times10^{%d}$" % (t / 10 ** exponent, exponent))
    ax.set_yticklabels(labels)
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def by_series(entries):
    groups = defaultdict(list)
    for e in entries:
        groups[series_name(e)].append(e)
    return groups


def log2_axis(ax, values):
    ax.set_xscale("log", base=2)
    ax.set_xticks(values)
    ax.set_xticklabels([str(v) for v in values])
    ax.minorticks_off()


# ----------------------------------------------------------------------------- figures

def plot_serial(summary, out):
    """Experiment 6: GFLOP/s of the serial kernels, grouped by N."""
    entries = [e for e in summary if e["mode"] == "serial"]
    if not entries:
        return
    sizes = sorted({e["N"] for e in entries})
    order = ["serial ijk", "serial ikj"] + sorted({series_name(e) for e in entries if e["kernel"] == "blocked"},
                                                  key=lambda s: int(re.search(r"\d+", s).group()))
    order = [s for s in order if any(series_name(e) == s for e in entries)]
    lookup = {(series_name(e), e["N"]): e for e in entries}

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    width = 0.8 / len(order)
    for i, name in enumerate(order):
        xs, ys, lo, hi = [], [], [], []
        for j, n in enumerate(sizes):
            e = lookup.get((name, n))
            if not e:
                continue
            g = e["gflops"]
            xs.append(j - 0.4 + width * (i + 0.5))
            ys.append(g)
            lo.append(g - 2 * n ** 3 / e["t_total_max"] / 1e9)
            hi.append(2 * n ** 3 / e["t_total_min"] / 1e9 - g)
        # The serial kernels are their own set of entities: slots in fixed order
        bars = ax.bar(xs, ys, width=width, color=SLOTS[i % len(SLOTS)],
                      edgecolor=SURFACE, linewidth=2, label=name.replace("serial ", ""))
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=TEXT_SECONDARY, elinewidth=1, capsize=2)
        for bar, y, top in zip(bars, ys, hi):
            ax.text(bar.get_x() + bar.get_width() / 2, y + top, f"{y:.1f}", ha="center", va="bottom",
                    fontsize=7.5, color=TEXT)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"N = {n}" for n in sizes])
    ax.set_ylabel("GFLOP/s (mediana; barra: mín.–máx.)")
    ax.set_title("Kernels seriais: desempenho por tamanho de matriz")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=len(order), loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8.5)
    finish(fig, os.path.join(out, "serial_kernels.png"))


def plot_correctness(summary, out):
    """Experiment 1: err_rel against N, with the sqrt(N) reference."""
    entries = [e for e in summary if e["err_rel"] is not None]
    if not entries:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    curves = {
        "serial / mpi (mesmo erro)": [e for e in entries if e["mode"] == "serial"],
        "mpi_blas": [e for e in entries if e["mode"] == "mpi_blas" and e["nprocs"] == 1],
    }
    color_of = {"serial / mpi (mesmo erro)": COLORS("mpi"), "mpi_blas": COLORS("mpi_blas")}
    for name, points in curves.items():
        points = sorted(points, key=lambda e: e["N"])
        if not points:
            continue
        xs = [e["N"] for e in points]
        ys = [e["err_rel"] for e in points]
        ax.plot(xs, ys, marker="o", color=color_of[name], label=name)
        if name.startswith("serial") and len(xs) > 1:
            ref = [ys[0] * math.sqrt(x / xs[0]) for x in xs]
            ax.plot(xs, ref, linestyle="--", linewidth=1.2, color=REFERENCE, zorder=3,
                    label="∝ √N (teoria; coincide com a curva azul)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    sizes = sorted({e["N"] for e in entries})
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.minorticks_off()
    log_ticks(ax, [e["err_rel"] for e in entries])
    ax.set_xlabel("N")
    ax.set_ylabel("erro relativo de Frobenius")
    ax.set_title("Erro da multiplicação em float (tolerância: 10⁻⁴)")
    ax.legend(loc="center right", fontsize=8.5)
    finish(fig, os.path.join(out, "correctness_error.png"))


def strong_series(entries):
    """Parallel series of the strong-scaling experiment (mpi with each b, mpi_blas)."""
    return {name: sorted(points, key=lambda e: e["nprocs"])
            for name, points in by_series(entries).items() if not name.startswith("serial")}


def plot_strong(summary, out):
    """Experiment 2: time, speedup and efficiency against P, one panel per N."""
    entries = [e for e in summary if e["mode"] != "serial"]
    if not entries:
        return
    sizes = sorted({e["N"] for e in entries})

    for metric in ("time", "speedup", "efficiency"):
        fig, axes = plt.subplots(1, len(sizes), figsize=(5.2 * len(sizes), 4.2), squeeze=False)
        for ax, n in zip(axes[0], sizes):
            series = strong_series([e for e in entries if e["N"] == n])
            procs = sorted({e["nprocs"] for s in series.values() for e in s})
            for name, points in series.items():
                base = next((e["t_total"] for e in points if e["nprocs"] == 1), None)
                xs = [e["nprocs"] for e in points]
                if metric == "time":
                    ys = [e["t_total"] for e in points]
                    err = [[e["t_total"] - e["t_total_min"] for e in points],
                           [e["t_total_max"] - e["t_total"] for e in points]]
                    ax.errorbar(xs, ys, yerr=err, marker="o", color=COLORS(name), capsize=2, elinewidth=1,
                                label=name)
                else:
                    if base is None:
                        continue
                    speedup = [base / e["t_total"] for e in points]
                    ys = speedup if metric == "speedup" else [s / p for s, p in zip(speedup, xs)]
                    ax.plot(xs, ys, marker="o", color=COLORS(name), label=name)
            log2_axis(ax, procs)
            if metric == "time":
                ax.set_yscale("log")
                ax.set_ylabel("t_total (s, mediana; barra: mín.–máx.)")
            elif metric == "speedup":
                ax.plot(procs, procs, linestyle="--", linewidth=1.2, color=REFERENCE, label="ideal")
                ax.set_yscale("log", base=2)
                ax.set_yticks(procs)
                ax.set_yticklabels([str(p) for p in procs])
                ax.set_ylabel("speedup  T(1) / T(P)")
            else:
                ax.axhline(1.0, linestyle="--", linewidth=1.2, color=REFERENCE, label="ideal")
                ax.set_ylim(0, 1.15)
                ax.set_ylabel("eficiência  S(P) / P")
            ax.set_xlabel("P (processos MPI)")
            ax.set_title(f"N = {n}")
            ax.legend(fontsize=8.5)
        title = {"time": "Escalabilidade forte: tempo", "speedup": "Escalabilidade forte: speedup",
                 "efficiency": "Escalabilidade forte: eficiência"}[metric]
        fig.suptitle(title, fontsize=12)
        finish(fig, os.path.join(out, f"strong_{metric}.png"))

    plot_breakdown(entries, out)


def plot_breakdown(entries, out):
    """Share of t_dist, t_comp and t_gather in t_total against P, for the largest N."""
    n = max(e["N"] for e in entries)
    series = strong_series([e for e in entries if e["N"] == n])
    names = [s for s in ("mpi (b=64)", "mpi (b=128)", "mpi_blas (sgemm único)") if s in series] or list(series)
    phases = [("t_dist", "distribuição (Bcast + Scatterv)", SLOTS[1]),
              ("t_comp", "conta local", SLOTS[0]),
              ("t_gather", "coleta (Gatherv)", SLOTS[2])]

    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 4.2), squeeze=False)
    for ax, name in zip(axes[0], names):
        points = series[name]
        labels = [str(e["nprocs"]) for e in points]
        bottom = [0.0] * len(points)
        for key, label, color in phases:
            share = [100 * e[key] / (e["t_dist"] + e["t_comp"] + e["t_gather"]) for e in points]
            ax.bar(labels, share, bottom=bottom, color=color, edgecolor=SURFACE, linewidth=2, label=label)
            bottom = [b + s for b, s in zip(bottom, share)]
        ax.set_ylim(0, 100)
        ax.set_xlabel("P (processos MPI)")
        ax.set_ylabel("% do t_total")
        ax.set_title(name)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.5)
    fig.suptitle(f"Para onde vai o tempo (N = {n})", fontsize=12)
    finish(fig, os.path.join(out, "strong_breakdown.png"), rect=(0, 0.07, 1, 1))


def plot_blocks(summary, out):
    """Experiment 3: GFLOP/s against the block size b, one panel per P."""
    entries = [e for e in summary if e["mode"] in ("mpi", "mpi_blas")]
    if not entries:
        return
    procs = sorted({e["nprocs"] for e in entries})
    fig, axes = plt.subplots(1, len(procs), figsize=(4.6 * len(procs), 4.2), squeeze=False)
    for ax, p in zip(axes[0], procs):
        points = [e for e in entries if e["nprocs"] == p]
        blocks = sorted({e["block"] for e in points if e["block"] < e["N"]})
        positions = {b: i for i, b in enumerate(blocks)}
        whole = len(blocks)
        for mode in ("mpi", "mpi_blas"):
            mode_points = sorted((e for e in points if e["mode"] == mode), key=lambda e: e["block"])
            xs = [positions.get(e["block"], whole) for e in mode_points]
            ys = [e["gflops"] for e in mode_points]
            if xs:
                ax.plot(xs, ys, marker="o", color=COLORS(mode), label=mode)
        ax.set_xticks(list(range(whole + 1)))
        ax.set_xticklabels([str(b) for b in blocks] + ["N"])
        ax.set_xlabel("tamanho do bloco b")
        ax.set_ylabel("GFLOP/s (total, mediana)")
        ax.set_title(f"P = {p}")
        ax.legend(fontsize=8.5)
    n = entries[0]["N"]
    fig.suptitle(f"Efeito do tamanho do bloco (N = {n}; \"N\" = um sgemm só)", fontsize=12)
    finish(fig, os.path.join(out, "blocks_gflops.png"))


def plot_sizes(summary, out):
    """Experiment 4: time against N (log-log) with the N^3 reference, one panel per P."""
    entries = [e for e in summary if e["mode"] != "serial"]
    if not entries:
        return
    procs = sorted({e["nprocs"] for e in entries})
    fig, axes = plt.subplots(1, len(procs), figsize=(4.6 * len(procs), 4.2), squeeze=False)
    for ax, p in zip(axes[0], procs):
        series = by_series([e for e in entries if e["nprocs"] == p])
        sizes = sorted({e["N"] for s in series.values() for e in s})
        for name, points in series.items():
            points = sorted(points, key=lambda e: e["N"])
            ax.plot([e["N"] for e in points], [e["t_total"] for e in points], marker="o",
                    color=COLORS(name), label=name)
        anchor = min((s for s in series.values()), key=lambda s: s[0]["t_total"])
        anchor = sorted(anchor, key=lambda e: e["N"])[0]
        ax.plot(sizes, [anchor["t_total"] * (n / anchor["N"]) ** 3 for n in sizes], linestyle="--",
                linewidth=1.2, color=REFERENCE, label="∝ N³")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(n) for n in sizes])
        ax.minorticks_off()
        ax.set_xlabel("N")
        ax.set_ylabel("t_total (s, mediana)")
        ax.set_title(f"P = {p}")
        ax.legend(fontsize=8)
    fig.suptitle("Efeito do tamanho da matriz", fontsize=12)
    finish(fig, os.path.join(out, "sizes_time.png"))


def plot_weak(summary, out):
    """Experiment 5: weak-scaling efficiency T(1, N1) / T(P, N_P)."""
    entries = [e for e in summary if e["mode"] != "serial"]
    if not entries:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    procs = sorted({e["nprocs"] for e in entries})
    for name, points in by_series(entries).items():
        points = sorted(points, key=lambda e: e["nprocs"])
        base = next((e["t_total"] for e in points if e["nprocs"] == 1), None)
        if base is None:
            continue
        ax.plot([e["nprocs"] for e in points], [base / e["t_total"] for e in points], marker="o",
                color=COLORS(name), label=name)
    ax.axhline(1.0, linestyle="--", linewidth=1.2, color=REFERENCE, label="ideal")
    log2_axis(ax, procs)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("P (processos MPI; N cresce com ∛P)")
    ax.set_ylabel("eficiência fraca  T(1, N₁) / T(P, N_P)")
    ax.set_title("Escalabilidade fraca (trabalho por processo constante)")
    ax.legend(fontsize=8.5)
    finish(fig, os.path.join(out, "weak_efficiency.png"))


def plot_hybrid(summary, out):
    """Experiment 8: time of each P x T combination (always 128 cores)."""
    entries = sorted(summary, key=lambda e: -e["nprocs"])
    if not entries:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    labels = [f"{e['nprocs']}×{e['blas_threads']}" for e in entries]
    values = [e["t_total"] for e in entries]
    bars = ax.bar(labels, values, color=COLORS("mpi_blas"), edgecolor=SURFACE, linewidth=2)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8, color=TEXT)
    cores = entries[0]["nprocs"] * entries[0]["blas_threads"]
    ax.set_xlabel(f"processos MPI × threads do OpenBLAS ({cores} núcleos)")
    ax.set_ylabel("t_total (s, mediana)")
    ax.set_title(f"mpi_blas híbrido (N = {entries[0]['N']})")
    ax.grid(axis="x", visible=False)
    finish(fig, os.path.join(out, "hybrid_time.png"))


# ----------------------------------------------------------------------------- main

PLOTS = {
    "correctness": plot_correctness,
    "serial": plot_serial,
    "strong": plot_strong,
    "blocks": plot_blocks,
    "sizes": plot_sizes,
    "weak": plot_weak,
    "hybrid": plot_hybrid,
}

TABLE_COLUMNS = ["mode", "kernel", "N", "block", "nprocs", "blas_threads", "reps",
                 "t_total", "t_total_min", "t_total_max", "t_dist", "t_comp", "t_gather",
                 "gflops", "t_verif", "err_rel", "ok"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results", help="directory with the <experiment>_<jobid>.csv files")
    parser.add_argument("--out", default=os.path.join("analysis", "figures"), help="output directory")
    args = parser.parse_args()

    setup_style()
    os.makedirs(args.out, exist_ok=True)
    experiments = load(args.results)
    if not experiments:
        print(f"No <experiment>_<jobid>.csv files in {args.results}")
        return

    for name, plot in PLOTS.items():
        rows = experiments.get(name)
        if not rows:
            print(f"{name}: no data, skipped")
            continue
        summary = aggregate(rows)
        print(f"{name}: {len(rows)} rows, {len(summary)} configurations")
        table = os.path.join(args.out, f"summary_{name}.csv")
        write_table(summary, table, TABLE_COLUMNS)
        print(f"  table:  {table}")
        plot(summary, args.out)


if __name__ == "__main__":
    main()
