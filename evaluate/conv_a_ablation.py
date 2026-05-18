"""Fixed Conv_a ablation for ConvPsoSwarm.

The experiment resets the NumPy RNG to the same seed before each optimizer
run. This makes each Conv_a setting start from the same initial population and
use the same r1/r2 random tensors as the PSO baselines for that trial.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from functions import CEC_functions
from matAgent.ccpso import ConvPsoSwarm
from matAgent.pso import PsoSwarm


DEFAULT_CONV_A_VALUES = (0.3, 0.8, 1.0, 1.3, 1.5, 1.8)
CLERC_W = 0.729844
CLERC_C1 = 1.496180
CLERC_C2 = 1.496180


class ClercPsoSwarm(PsoSwarm):
    """Standard velocity PSO with the same fixed coefficients as ConvPSO."""

    optimizer_name = "Clerc_PSO"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.optimizer_name

    def run_once(self, actions=None):
        self.r1 = np.random.uniform(0, 1, (self.n_part, self.n_dim))
        self.r2 = np.random.uniform(0, 1, (self.n_part, self.n_dim))
        self.vs = (
            CLERC_W * self.vs
            + CLERC_C1 * self.r1 * (self.p_best - self.xs)
            + CLERC_C2 * self.r2 * (self.history_best_x - self.xs)
        )
        self.vs = np.clip(self.vs, self.min_v, self.max_v)
        self.xs = np.clip(self.xs + self.vs, self.pos_min, self.pos_max)
        self.fits = self.fun(self.xs)
        self.update_best()


def parse_conv_a_values(raw):
    values = []
    for value in raw.split(","):
        value = value.strip()
        if value:
            values.append(float(value))
    if not values:
        raise ValueError("At least one Conv_a value is required.")
    return values


def build_cec_fun(dim, f_num):
    cec_functions = CEC_functions(dim)

    def test_fun(x):
        return cec_functions.Y(x, f_num)

    return test_fun


def get_n_run(max_fe, n_part):
    if max_fe < n_part:
        raise ValueError("max_fe must be at least n_part.")
    if max_fe % n_part != 0:
        raise ValueError("max_fe must be divisible by n_part for full-swarm FE accounting.")
    return int((max_fe - n_part) / n_part)


def run_optimizer(optimizer_cls, seed, fun, args, config_dic=None):
    np.random.seed(seed)
    optimizer = optimizer_cls(
        n_run=args.n_run,
        n_part=args.n_part,
        show=False,
        fun=fun,
        n_dim=args.dim,
        pos_max=args.pos_max,
        pos_min=args.pos_min,
        config_dic=config_dic or {},
    )
    optimizer.run()
    return {
        "final_best": float(optimizer.history_best_fit),
        "final_fe": int(optimizer.fe_num),
        "record_points": len(optimizer.result_cache),
    }


def collect_rows(args):
    fun = build_cec_fun(args.dim, args.f_num)
    rows = []

    for trial in range(args.trials):
        seed = args.seed_base + trial
        trial_specs = [
            ("PSO", None, PsoSwarm, {}),
        ]
        if args.include_clerc_pso:
            trial_specs.append(("Clerc_PSO", None, ClercPsoSwarm, {}))
        for conv_a in args.conv_a_values:
            trial_specs.append(
                (
                    f"Conv_PSO_fixed_{conv_a:g}",
                    conv_a,
                    ConvPsoSwarm,
                    {"fixed_conv_a": conv_a},
                )
            )

        for algorithm, conv_a, optimizer_cls, config_dic in trial_specs:
            result = run_optimizer(optimizer_cls, seed, fun, args, config_dic=config_dic)
            rows.append(
                {
                    "f_num": args.f_num,
                    "dim": args.dim,
                    "max_fe": args.max_fe,
                    "n_part": args.n_part,
                    "trial": trial,
                    "seed": seed,
                    "algorithm": algorithm,
                    "conv_a": "" if conv_a is None else conv_a,
                    "final_best": result["final_best"],
                    "final_fe": result["final_fe"],
                    "record_points": result["record_points"],
                }
            )
            print(
                f"trial={trial:03d} seed={seed} algorithm={algorithm:<18} "
                f"best={result['final_best']:.12g}"
            )

    return rows


def summarize_rows(rows):
    grouped = {}
    pso_by_seed = {}
    for row in rows:
        if row["algorithm"] == "PSO":
            pso_by_seed[row["seed"]] = row["final_best"]

        key = (row["algorithm"], row["conv_a"])
        grouped.setdefault(key, []).append(row)

    summaries = []
    for (algorithm, conv_a), group_rows in grouped.items():
        values = np.asarray([row["final_best"] for row in group_rows], dtype=float)
        wins = []
        for row in group_rows:
            pso_value = pso_by_seed.get(row["seed"])
            if pso_value is not None and row["algorithm"] != "PSO":
                wins.append(float(row["final_best"] < pso_value))

        summaries.append(
            {
                "algorithm": algorithm,
                "conv_a": conv_a,
                "runs": len(group_rows),
                "mean_final_best": float(np.mean(values)),
                "std_final_best": float(np.std(values)),
                "median_final_best": float(np.median(values)),
                "min_final_best": float(np.min(values)),
                "max_final_best": float(np.max(values)),
                "win_rate_vs_pso": "" if not wins else float(np.mean(wins)),
            }
        )

    summaries.sort(key=lambda item: item["mean_final_best"])
    for rank, item in enumerate(summaries, start=1):
        item["rank_by_mean"] = rank
    return summaries


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(summary_rows, output_path, title):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skip plot generation.")
        return None

    conv_rows = [
        row for row in summary_rows
        if str(row["algorithm"]).startswith("Conv_PSO_fixed_")
    ]
    conv_rows.sort(key=lambda row: float(row["conv_a"]))
    if not conv_rows:
        return None

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    x = [float(row["conv_a"]) for row in conv_rows]
    y = [float(row["mean_final_best"]) for row in conv_rows]
    yerr = [float(row["std_final_best"]) for row in conv_rows]
    ax.errorbar(x, y, yerr=yerr, marker="o", capsize=4, label="Fixed ConvPSO")

    for baseline_name, color in (("PSO", "#555555"), ("Clerc_PSO", "#1f77b4")):
        baseline = next((row for row in summary_rows if row["algorithm"] == baseline_name), None)
        if baseline:
            ax.axhline(
                float(baseline["mean_final_best"]),
                linestyle="--",
                linewidth=1.2,
                color=color,
                label=f"{baseline_name} mean",
            )

    ax.set_title(title)
    ax.set_xlabel("Fixed Conv_a")
    ax.set_ylabel("Final best fitness (lower is better)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def print_summary(summary_rows):
    print("\nSummary: final best fitness, lower is better")
    header = (
        f"{'rank':>4}  {'algorithm':<20} {'Conv_a':>8} {'runs':>5} "
        f"{'mean':>16} {'std':>12} {'win_vs_pso':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        conv_a = row["conv_a"] if row["conv_a"] != "" else "-"
        win = row["win_rate_vs_pso"] if row["win_rate_vs_pso"] != "" else "-"
        mean = row["mean_final_best"]
        std = row["std_final_best"]
        print(
            f"{row['rank_by_mean']:>4}  {row['algorithm']:<20} {str(conv_a):>8} "
            f"{row['runs']:>5} {mean:>16.8g} {std:>12.5g} {str(win):>10}"
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run fixed Conv_a ablation on CEC Rastrigin by default."
    )
    parser.add_argument("--f-num", type=int, default=11, help="CEC function number; 11 is Rastrigin.")
    parser.add_argument("--dim", type=int, default=30)
    parser.add_argument("--n-part", type=int, default=100)
    parser.add_argument("--max-fe", type=int, default=10000)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=20260519)
    parser.add_argument(
        "--conv-a-values",
        type=parse_conv_a_values,
        default=list(DEFAULT_CONV_A_VALUES),
        help="Comma-separated fixed Conv_a values, e.g. 0.3,0.8,1.0,1.3,1.5,1.8.",
    )
    parser.add_argument("--pos-min", type=float, default=-100.0)
    parser.add_argument("--pos-max", type=float, default=100.0)
    parser.add_argument("--output-dir", type=Path, default=Path("data") / "ablation")
    parser.add_argument("--no-clerc-pso", action="store_true", help="Do not run the same-coefficient PSO baseline.")
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.conv_a_values = list(args.conv_a_values)
    args.include_clerc_pso = not args.no_clerc_pso
    args.n_run = get_n_run(args.max_fe, args.n_part)

    started_at = time.strftime("%Y%m%d_%H%M%S")
    stem = (
        f"conv_a_ablation_f{args.f_num}_d{args.dim}_"
        f"fe{args.max_fe}_runs{args.trials}_{started_at}"
    )
    raw_path = args.output_dir / f"{stem}_raw.csv"
    summary_path = args.output_dir / f"{stem}_summary.csv"
    plot_path = args.output_dir / f"{stem}.png"

    print(
        "Running fixed Conv_a ablation: "
        f"f_num={args.f_num}, dim={args.dim}, max_fe={args.max_fe}, "
        f"n_part={args.n_part}, trials={args.trials}, "
        f"Conv_a={args.conv_a_values}"
    )
    rows = collect_rows(args)
    summary_rows = summarize_rows(rows)

    write_csv(
        raw_path,
        rows,
        [
            "f_num",
            "dim",
            "max_fe",
            "n_part",
            "trial",
            "seed",
            "algorithm",
            "conv_a",
            "final_best",
            "final_fe",
            "record_points",
        ],
    )
    write_csv(
        summary_path,
        summary_rows,
        [
            "rank_by_mean",
            "algorithm",
            "conv_a",
            "runs",
            "mean_final_best",
            "std_final_best",
            "median_final_best",
            "min_final_best",
            "max_final_best",
            "win_rate_vs_pso",
        ],
    )

    generated_plot = None
    if not args.no_plot:
        generated_plot = maybe_plot(
            summary_rows,
            plot_path,
            title=f"Fixed Conv_a Ablation on CEC F{args.f_num}",
        )

    print_summary(summary_rows)
    print(f"\nRaw results: {raw_path}")
    print(f"Summary: {summary_path}")
    if generated_plot:
        print(f"Plot: {generated_plot}")


if __name__ == "__main__":
    main()
