import csv
import os
import pickle
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Fix missing CJK fonts on Linux/WSL.
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

DEFAULT_OUTPUT_DIR = "final_battle_plots"
ERROR_FLOOR = 1e-12

CEC2013_OPTIMUM = {
    1: -1400,
    2: -1300,
    3: -1200,
    4: -1100,
    5: -1000,
    6: -900,
    7: -800,
    8: -700,
    9: -600,
    10: -500,
    11: -400,
    12: -300,
    13: -200,
    14: -100,
    15: 100,
    16: 200,
    17: 300,
    18: 400,
    19: 500,
    20: 600,
    21: 700,
    22: 800,
    23: 900,
    24: 1000,
    25: 1100,
    26: 1200,
    27: 1300,
    28: 1400,
}


OPTIMIZER_LABEL_MAP = {
    "PSOorigin": "PSO",
    "PSOtrain": "RLPSO",
    "PSO-train": "RLPSO",
    "RLPSO-train": "RLPSO",
    "Stage1-RL+BasicPSO-train": "RLPSO",
    "Conv_PSOtrain": "RL_CCPSO",
    "Conv_PSO_DualCtrain": "RL_CCPSO",
    "RLCCPSO-train": "RL_CCPSO",
    "Stage2-RL+BasicPSO+Convergence-train": "RL_CCPSO",
}


OPTIMIZER_COLOR_MAP = {
    'RL_CCPSO': '#e41a1c',
    'RLPSO': '#377eb8',
    'PSO': '#4daf4a',
}


OPTIMIZER_MARKER_MAP = {
    'RL_CCPSO': 'o',
    'RLPSO': 's',
    'PSO': '^',
}


def _display_optimizer_label(opt_name):
    text = str(opt_name)
    if text in OPTIMIZER_LABEL_MAP:
        return OPTIMIZER_LABEL_MAP[text]
    return text


def _style_base_label(label):
    text = str(label)
    for base_label in ('RLPSO', 'PSO', 'RL_CCPSO'):
        if text == base_label or text.startswith(f"{base_label} "):
            return base_label
    return text


def _optimizer_color(label, fallback='#999999'):
    return OPTIMIZER_COLOR_MAP.get(_style_base_label(label), fallback)


def _optimizer_marker(label, fallback='x'):
    return OPTIMIZER_MARKER_MAP.get(_style_base_label(label), fallback)


class MockClass:
    pass


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return type(name, (MockClass,), {"__module__": module})


def _get_pickle_path(task_md5):
    return os.path.join("data", "task", str(task_md5), "result.pickle")


def _load_task_result(task_md5):
    pickle_path = _get_pickle_path(task_md5)
    if not os.path.exists(pickle_path):
        print(f"未找到结果文件: {pickle_path}")
        return None

    with open(pickle_path, "rb") as f:
        return SafeUnpickler(f).load()


def _normalize_function_ids(functions):
    normalized = []
    for fun in functions:
        try:
            normalized.append(int(fun))
        except (TypeError, ValueError):
            normalized.append(fun)
    return normalized


def _get_target_functions(summary_result, target_functions=None):
    if target_functions is not None:
        return _normalize_function_ids(target_functions)

    functions = summary_result.get("functions")
    if functions:
        return _normalize_function_ids(functions)

    result_keys = list(summary_result.get("result", {}).keys())
    normalized_keys = _normalize_function_ids(result_keys)
    try:
        return sorted(normalized_keys)
    except TypeError:
        return normalized_keys


def _get_cec2013_optimum(fun_num):
    try:
        return CEC2013_OPTIMUM[int(fun_num)]
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"Unknown CEC2013 function id: {fun_num}")


def _fitness_error(values, f_opt):
    errors = np.asarray(values, dtype=float) - float(f_opt)
    return np.maximum(errors, ERROR_FLOOR)


def _summarize_conv_runs(conv_runs):
    if not conv_runs:
        return None

    fe_value_map = {}
    for run_trace in conv_runs:
        for fe, conv_a in run_trace:
            fe = int(fe)
            conv_a = float(conv_a)
            if fe not in fe_value_map:
                fe_value_map[fe] = []
            fe_value_map[fe].append(conv_a)

    fe_points = sorted(fe_value_map.keys())
    mean_vals = []
    std_vals = []
    var_vals = []

    for fe in fe_points:
        values = np.asarray(fe_value_map[fe], dtype=float)
        mean_vals.append(float(np.mean(values)))
        std_vals.append(float(np.std(values)))
        var_vals.append(float(np.var(values)))

    return {
        'fe': fe_points,
        'mean': mean_vals,
        'std': std_vals,
        'var': var_vals,
    }


def _resolve_output_dir(output_dir=None):
    return output_dir or os.environ.get("FINAL_BATTLE_PLOT_DIR") or DEFAULT_OUTPUT_DIR


def extract_csv(task_md5, output_dir=None):
    obj = _load_task_result(task_md5)
    if obj is None:
        return

    summary = obj["result"][0]

    output_dir = _resolve_output_dir(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "final_summary.csv")

    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "function",
            "optimizer",
            "fe",
            "raw_mean",
            "raw_best",
            "std",
            "f_opt",
            "mean_error",
            "best_error",
        ])

        for fun_num, fun_res in summary["result"].items():
            f_opt = _get_cec2013_optimum(fun_num)
            for name, res in fun_res.items():
                fe, mean, best, std = res["result"][-1]
                mean_error = float(_fitness_error([mean], f_opt)[0])
                best_error = float(_fitness_error([best], f_opt)[0])
                writer.writerow([fun_num, name, fe, mean, best, std, f_opt, mean_error, best_error])

    print("average_ranks =", summary.get("average_ranks", "N/A"))
    print(f"saved -> {summary_path}")


def plot_highlight_functions(task_md5, target_functions=None, output_dir=None):
    data = _load_task_result(task_md5)
    if data is None:
        return

    output_dir = _resolve_output_dir(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("正在绘制收敛曲线...")
    summary = data['result'][0]
    real_results = summary['result']
    target_functions = _get_target_functions(summary, target_functions=target_functions)

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['#999999'])

    for f_num in target_functions:
        if f_num in real_results:
            opt_dicts = real_results[f_num]
        elif str(f_num) in real_results:
            opt_dicts = real_results[str(f_num)]
        else:
            continue

        f_opt = _get_cec2013_optimum(f_num)
        plt.figure(figsize=(10, 6))

        for opt_index, (opt_name, res_data) in enumerate(opt_dicts.items()):
            matrix = res_data['result']
            if matrix is None or len(matrix) == 0:
                continue

            label_name = _display_optimizer_label(opt_name)
            x_vals = [row[0] for row in matrix]
            raw_best = [row[2] if len(row) > 2 else row[-1] for row in matrix]
            y_vals = _fitness_error(raw_best, f_opt)

            color = _optimizer_color(label_name, color_cycle[opt_index % len(color_cycle)])
            marker = _optimizer_marker(label_name)
            mark_step = max(1, len(x_vals) // 15)

            plt.plot(
                x_vals,
                y_vals,
                label=label_name,
                color=color,
                marker=marker,
                markevery=mark_step,
                linewidth=1.5,
                zorder=5,
                alpha=0.9,
            )

        plt.title(f"CEC2013 F{f_num} Convergence by Optimality Gap", fontsize=15, fontweight='bold')
        plt.xlabel("Function Evaluations (FEs)", fontsize=13)
        plt.ylabel(r"Fitness Error: f(x) - f(x*)(log)", fontsize=13)

        ax = plt.gca()
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=8))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10))
        ax.text(
            0.02,
            0.02,
            rf"$f^*={f_opt:g}$, errors clipped at {ERROR_FLOOR:g}",
            transform=ax.transAxes,
            fontsize=9,
            color="#555555",
            ha="left",
            va="bottom",
        )

        plt.legend(fontsize=11, loc='best', framealpha=0.8)
        plt.grid(True, linestyle='--', alpha=0.5)

        save_path = os.path.join(output_dir, f"F{f_num}_convergence_academic.png")
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close()

        print(f"✅ F{f_num} 的收敛曲线已保存至 -> {save_path}")


def plot_conv_a_traces(task_md5, target_functions=None, output_dir=None):
    data = _load_task_result(task_md5)
    if data is None:
        return

    output_dir = _resolve_output_dir(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    summary = data['result'][0]
    real_results = summary['result']
    target_functions = _get_target_functions(summary, target_functions=target_functions)

    for f_num in target_functions:
        opt_dicts = real_results.get(f_num) or real_results.get(str(f_num))
        if not opt_dicts:
            continue

        conv_items = [
            (opt_name, opt_res)
            for opt_name, opt_res in opt_dicts.items()
            if opt_res and opt_res.get("conv_runs")
        ]
        if not conv_items:
            continue

        color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get(
            'color',
            ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3'],
        )

        fig, (ax_trace, ax_var) = plt.subplots(
            2,
            1,
            figsize=(10, 9),
            sharex=True,
            gridspec_kw={'height_ratios': [3, 1.5]},
        )
        plotted = False

        for conv_index, (conv_key, conv_res) in enumerate(conv_items):
            conv_runs = conv_res.get("conv_runs", [])
            conv_stats = conv_res.get("conv_stats") or _summarize_conv_runs(conv_runs)
            if conv_stats is None:
                continue

            label = _display_optimizer_label(conv_key)
            color = _optimizer_color(label, color_cycle[conv_index % len(color_cycle)])
            fe_vals = np.asarray(conv_stats["fe"], dtype=float)
            mean_vals = np.asarray(conv_stats["mean"], dtype=float)
            std_vals = np.asarray(conv_stats["std"], dtype=float)
            var_vals = np.asarray(conv_stats["var"], dtype=float)

            if len(conv_items) == 1:
                for run_trace in conv_runs:
                    x_vals = [row[0] for row in run_trace]
                    y_vals = [row[1] for row in run_trace]
                    ax_trace.plot(x_vals, y_vals, color=color, alpha=0.25, linewidth=1.0)

            lower = np.clip(mean_vals - std_vals, 0.0, 2.0)
            upper = np.clip(mean_vals + std_vals, 0.0, 2.0)
            ax_trace.fill_between(
                fe_vals,
                lower,
                upper,
                color=color,
                alpha=0.16,
            )
            ax_trace.plot(fe_vals, mean_vals, color=color, linewidth=2.5, label=f"{label} mean")
            ax_var.plot(fe_vals, var_vals, color=color, linewidth=2.0, label=label)
            ax_var.fill_between(fe_vals, 0, var_vals, color=color, alpha=0.12)
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax_trace.set_title(f"Conv_a Mean and Variance on F{f_num}")
        ax_trace.set_ylabel("Conv_a")
        ax_trace.set_ylim(0, 2)
        ax_trace.grid(True, linestyle='--', alpha=0.5)
        ax_trace.legend(loc='best', framealpha=0.85)

        ax_var.set_xlabel("Function Evaluations (FEs)")
        ax_var.set_ylabel("Var")
        ax_var.grid(True, linestyle='--', alpha=0.5)
        ax_var.legend(loc='best', framealpha=0.85)

        save_path = os.path.join(output_dir, f"F{f_num}_conv_a_stats.png")
        fig.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ F{f_num} 的 Conv_a 均值/方差图已保存至 -> {save_path}")


def generate_all_plots(task_md5, output_dir=None):
    extract_csv(task_md5, output_dir=output_dir)
    plot_highlight_functions(task_md5, output_dir=output_dir)
    plot_conv_a_traces(task_md5, output_dir=output_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python plot_final_battle.py <task_md5> [output_dir]")

    cli_output_dir = sys.argv[2] if len(sys.argv) >= 3 else None
    generate_all_plots(sys.argv[1], output_dir=cli_output_dir)
