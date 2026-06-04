"""
Lightweight Q / pbest / gbest diagnosis for the current RLCCPSO mainline.

The CSV records raw CEC2013 F1 fitness values. For F1, the optimum is -1400, so
better values are closer to -1400 from above.
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from functions import CEC_functions
from matAgent.ccpso import ConvPsoSwarm
from task.experiment_config import EXPERIMENT_CCPSO_CONFIG


DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "diagnosis"


def build_cec2013_f1(dim):
    cec_functions = CEC_functions(dim)

    def f1(x):
        if len(x.shape) == 2:
            return np.asarray([cec_functions.Y(row, 1) for row in x], dtype=float)
        return np.asarray(cec_functions.Y(x, 1), dtype=float)

    return f1


def evaluate_without_fe(optimizer, points):
    return np.asarray(optimizer.fitness_valuate(np.asarray(points, dtype=float)), dtype=float)


def normalized_diversity(points, pos_min, pos_max):
    points = np.asarray(points, dtype=float)
    span = max(float(pos_max - pos_min), 1e-12)
    return float(np.mean(np.std(points, axis=0)) / span)


def normalized_mean_row_distance(left, right, dim, pos_min, pos_max):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    span = max(float(pos_max - pos_min), 1e-12)
    norm = max(np.sqrt(dim) * span, 1e-12)
    return float(np.mean(np.linalg.norm(left - right, axis=1)) / norm)


def add_vector_columns(row, prefix, vector):
    for index, value in enumerate(np.asarray(vector, dtype=float).reshape(-1)):
        row[f"{prefix}_d{index:02d}"] = float(value)


def coordinate_fieldnames(prefix, dim):
    return [f"{prefix}_d{index:02d}" for index in range(dim)]


def collect_row(run_id, optimizer):
    q = optimizer.current_q
    pbest = optimizer.current_pbest
    pbest_values = optimizer.current_pbest_fit
    gbest = optimizer.current_gbest
    gbest_value = optimizer.current_gbest_fit
    x_before_update = optimizer.current_x_before_update

    if q is None or pbest is None or pbest_values is None or gbest is None:
        raise RuntimeError("ConvPsoSwarm did not expose Q/pbest/gbest diagnosis fields.")

    q_values = evaluate_without_fe(optimizer, q)
    gbest_matrix = np.broadcast_to(gbest, q.shape)
    row = {
        "run": run_id,
        "fe": int(optimizer.fe_num),
        "q_mean_value": float(np.mean(q_values)),
        "q_best_value": float(np.min(q_values)),
        "q_std_value": float(np.std(q_values)),
        "pbest_mean_value": float(np.mean(pbest_values)),
        "pbest_best_value": float(np.min(pbest_values)),
        "pbest_std_value": float(np.std(pbest_values)),
        "gbest_value": float(gbest_value),
        "swarm_diversity": normalized_diversity(optimizer.xs, optimizer.pos_min, optimizer.pos_max),
        "q_diversity": normalized_diversity(q, optimizer.pos_min, optimizer.pos_max),
        "pbest_diversity": normalized_diversity(pbest, optimizer.pos_min, optimizer.pos_max),
        "mean_distance_q_gbest": normalized_mean_row_distance(
            q,
            gbest_matrix,
            optimizer.n_dim,
            optimizer.pos_min,
            optimizer.pos_max,
        ),
        "mean_distance_pbest_gbest": normalized_mean_row_distance(
            pbest,
            gbest_matrix,
            optimizer.n_dim,
            optimizer.pos_min,
            optimizer.pos_max,
        ),
        "mean_distance_x_q": normalized_mean_row_distance(
            x_before_update,
            q,
            optimizer.n_dim,
            optimizer.pos_min,
            optimizer.pos_max,
        ),
        "conv_a": float(optimizer.current_conv_a),
        "conv_a_base": float(optimizer.current_conv_a_base),
        "conv_a_delta": float(optimizer.current_conv_a_delta),
        "stagnation_boost": float(optimizer.current_stagnation_boost),
        "c1": float(optimizer.current_c1),
        "c2": float(optimizer.current_c2),
    }

    add_vector_columns(row, "q_centroid", np.mean(q, axis=0))
    add_vector_columns(row, "pbest_centroid", np.mean(pbest, axis=0))
    add_vector_columns(row, "gbest", gbest)
    return row


def build_optimizer(dim, max_fe, n_part, group, model):
    if max_fe < n_part:
        raise ValueError("max_fe must be at least n_part.")
    if max_fe % n_part != 0:
        raise ValueError("max_fe must be divisible by n_part for full-swarm diagnosis.")

    config = dict(EXPERIMENT_CCPSO_CONFIG)
    config.update({
        "group": group,
        "max_fes": max_fe,
    })
    if model:
        config["model"] = model

    n_run = int((max_fe - n_part) / n_part)
    return ConvPsoSwarm(
        n_run=n_run,
        n_part=n_part,
        show=False,
        fun=build_cec2013_f1(dim),
        n_dim=dim,
        pos_max=100,
        pos_min=-100,
        config_dic=config,
    )


def run_single_diagnosis(run_id, seed, dim, max_fe, n_part, group, model):
    np.random.seed(seed + run_id)
    optimizer = build_optimizer(dim, max_fe, n_part, group, model)

    rows = []
    while optimizer.step_num < optimizer.n_run and optimizer.run_flag:
        optimizer.step_num += 1

        actions = None
        if optimizer.ddpg_actor:
            action_tensor = optimizer.ddpg_actor.policy(optimizer.get_state())
            actions = action_tensor.numpy() if hasattr(action_tensor, "numpy") else action_tensor

        optimizer.run_once(actions=actions)
        rows.append(collect_row(run_id, optimizer))

    return rows


def write_variable_notes(path):
    notes = [
        "run: Independent run id.",
        "fe: Current function evaluation count.",
        "q_mean_value/q_best_value/q_std_value: Raw CEC2013 F1 fitness values of Q points.",
        "pbest_mean_value/pbest_best_value/pbest_std_value: Raw fitness values stored in pbest.",
        "gbest_value: Raw global best fitness value. The F1 optimum is -1400.",
        "swarm_diversity/q_diversity/pbest_diversity: Mean coordinate std normalized by search range.",
        "mean_distance_q_gbest: Mean normalized Euclidean distance from each Q_i to gbest.",
        "mean_distance_pbest_gbest: Mean normalized Euclidean distance from each pbest_i to gbest.",
        "mean_distance_x_q: Mean normalized Euclidean distance from x_i before update to Q_i.",
        "conv_a: Effective convergence coefficient used in this step.",
        "conv_a_base: Delayed progress-prior base before actor residual and stagnation boost.",
        "conv_a_delta: Actor residual contribution.",
        "stagnation_boost: Extra Conv_a added after no-improvement stagnation.",
        "c1/c2: Fixed Clerc PSO coefficients used to build Q.",
        "q_centroid_dXX: Coordinate XX of the centroid of all Q_i points.",
        "pbest_centroid_dXX: Coordinate XX of the centroid of all pbest_i points.",
        "gbest_dXX: Coordinate XX of the gbest point used when Q was calculated.",
    ]
    path.write_text("\n".join(notes) + "\n", encoding="utf-8")


def write_csv(path, rows, dim):
    fieldnames = [
        "run",
        "fe",
        "q_mean_value",
        "q_best_value",
        "q_std_value",
        "pbest_mean_value",
        "pbest_best_value",
        "pbest_std_value",
        "gbest_value",
        "swarm_diversity",
        "q_diversity",
        "pbest_diversity",
        "mean_distance_q_gbest",
        "mean_distance_pbest_gbest",
        "mean_distance_x_q",
        "conv_a",
        "conv_a_base",
        "conv_a_delta",
        "stagnation_boost",
        "c1",
        "c2",
    ]
    fieldnames.extend(coordinate_fieldnames("q_centroid", dim))
    fieldnames.extend(coordinate_fieldnames("pbest_centroid", dim))
    fieldnames.extend(coordinate_fieldnames("gbest", dim))

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose Q/pbest/gbest collapse on CEC2013 F1.")
    parser.add_argument("--runs", type=int, default=5, help="Independent runs; use 3 or 5 for small diagnosis.")
    parser.add_argument("--dim", type=int, default=30, help="CEC2013 F1 dimension.")
    parser.add_argument("--max-fe", type=int, default=10000, help="Maximum function evaluations.")
    parser.add_argument("--n-part", type=int, default=100, help="Particle number.")
    parser.add_argument("--group", type=int, default=1, help="RL action group number.")
    parser.add_argument("--seed", type=int, default=20260527, help="Base random seed.")
    parser.add_argument("--model", default=None, help="Optional trained actor .h5 path. Omit to use action=0.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for diagnosis CSV.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for run_id in range(args.runs):
        all_rows.extend(
            run_single_diagnosis(
                run_id=run_id,
                seed=args.seed,
                dim=args.dim,
                max_fe=args.max_fe,
                n_part=args.n_part,
                group=args.group,
                model=args.model,
            )
        )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = output_dir / f"q_pbest_gbest_f1_rlccpso_d{args.dim}_fe{args.max_fe}_runs{args.runs}_{timestamp}.csv"
    notes_path = csv_path.with_suffix(".variables.txt")
    write_csv(csv_path, all_rows, args.dim)
    write_variable_notes(notes_path)

    print(f"saved csv: {csv_path}")
    print(f"saved notes: {notes_path}")


if __name__ == "__main__":
    main()
