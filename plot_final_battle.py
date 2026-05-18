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


def extract_csv(task_md5):
    obj = _load_task_result(task_md5)
    if obj is None:
        return

    summary = obj["result"][0]

    with open("final_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["function", "optimizer", "fe", "mean", "best", "std"])

        for fun_num, fun_res in summary["result"].items():
            for name, res in fun_res.items():
                fe, mean, best, std = res["result"][-1]
                writer.writerow([fun_num, name, fe, mean, best, std])

    print("average_ranks =", summary.get("average_ranks", "N/A"))
    print("saved -> final_summary.csv")


def plot_highlight_functions(task_md5, target_functions=None):
    data = _load_task_result(task_md5)
    if data is None:
        return

    output_dir = "final_battle_plots"
    os.makedirs(output_dir, exist_ok=True)

    print("正在绘制收敛曲线...")
    summary = data['result'][0]
    real_results = summary['result']
    target_functions = _get_target_functions(summary, target_functions=target_functions)

    label_map = {
        "PSOorigin": "PSO",
        "PSOtrain": "RLPSO",
        "Conv_PSOtrain": "RL_CCPSO",
    }

    colors = {
        'RL_CCPSO': '#e41a1c',
        'RLPSO': '#377eb8',
        'PSO': '#4daf4a',
    }

    markers = {
        'RL_CCPSO': 'o',
        'RLPSO': 's',
        'PSO': '^',
    }

    for f_num in target_functions:
        if f_num in real_results:
            opt_dicts = real_results[f_num]
        elif str(f_num) in real_results:
            opt_dicts = real_results[str(f_num)]
        else:
            continue

        plt.figure(figsize=(10, 6))

        all_y = []
        for _, res_data in opt_dicts.items():
            matrix = res_data['result']
            if matrix is None or len(matrix) == 0:
                continue
            all_y.extend([row[2] if len(row) > 2 else row[-1] for row in matrix])

        all_y = np.array(all_y)
        if all_y.size > 0:
            abs_y = np.abs(all_y)
            nonzero_abs_y = abs_y[abs_y > 0]
            linthresh = max(nonzero_abs_y.min(), 1e-12) if nonzero_abs_y.size > 0 else 1e-8
        else:
            linthresh = 1e-8

        for opt_name, res_data in opt_dicts.items():
            matrix = res_data['result']
            if matrix is None or len(matrix) == 0:
                continue

            label_name = label_map.get(opt_name, str(opt_name))
            x_vals = [row[0] for row in matrix]
            y_vals = [row[2] if len(row) > 2 else row[-1] for row in matrix]

            color = colors.get(label_name, '#999999')
            marker = markers.get(label_name, 'x')
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

        plt.title(f"Convergence Curves on 30D Complex Function F{f_num}", fontsize=15, fontweight='bold')
        plt.xlabel("Function Evaluations (FEs)", fontsize=13)
        plt.ylabel("Fitness Value (Log Scale)", fontsize=13)

        ax = plt.gca()
        ax.set_yscale('symlog', linthresh=linthresh)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))

        def y_formatter(val, pos):
            if val == 0:
                return "0"
            if abs(val) < 1e-3 or abs(val) >= 1e4:
                return f"{val:.1e}"
            return f"{val:.4g}"

        ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_formatter))

        plt.legend(fontsize=11, loc='best', framealpha=0.8)
        plt.grid(True, linestyle='--', alpha=0.5)

        save_path = os.path.join(output_dir, f"F{f_num}_convergence_academic.png")
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close()

        print(f"✅ F{f_num} 的收敛曲线已保存至 -> {save_path}")


def plot_conv_a_traces(task_md5, target_functions=None):
    data = _load_task_result(task_md5)
    if data is None:
        return

    output_dir = "final_battle_plots"
    os.makedirs(output_dir, exist_ok=True)

    summary = data['result'][0]
    real_results = summary['result']
    target_functions = _get_target_functions(summary, target_functions=target_functions)

    for f_num in target_functions:
        opt_dicts = real_results.get(f_num) or real_results.get(str(f_num))
        if not opt_dicts:
            continue

        conv_key = next(
            (
                opt_name for opt_name in opt_dicts.keys()
                if str(opt_name).startswith("Conv_PSO") and str(opt_name).endswith("train")
            ),
            None,
        )
        if conv_key is None:
            continue

        conv_res = opt_dicts.get(conv_key)
        if not conv_res:
            continue

        conv_runs = conv_res.get("conv_runs", [])
        if not conv_runs:
            continue

        conv_stats = conv_res.get("conv_stats") or _summarize_conv_runs(conv_runs)
        if conv_stats is None:
            continue

        fe_vals = np.asarray(conv_stats["fe"], dtype=float)
        mean_vals = np.asarray(conv_stats["mean"], dtype=float)
        std_vals = np.asarray(conv_stats["std"], dtype=float)
        var_vals = np.asarray(conv_stats["var"], dtype=float)

        fig, (ax_trace, ax_var) = plt.subplots(
            2,
            1,
            figsize=(10, 9),
            sharex=True,
            gridspec_kw={'height_ratios': [3, 1.5]},
        )

        for run_trace in conv_runs:
            x_vals = [row[0] for row in run_trace]
            y_vals = [row[1] for row in run_trace]
            ax_trace.plot(x_vals, y_vals, color="#f26c6c", alpha=0.25, linewidth=1.0)

        lower = np.clip(mean_vals - std_vals, 0.0, 2.0)
        upper = np.clip(mean_vals + std_vals, 0.0, 2.0)
        ax_trace.fill_between(
            fe_vals,
            lower,
            upper,
            color="#e41a1c",
            alpha=0.18,
            label="mean ± std",
        )
        ax_trace.plot(fe_vals, mean_vals, color="#c00000", linewidth=2.5, label="mean Conv_a")
        ax_trace.set_title(f"Conv_a Mean and Variance on F{f_num}")
        ax_trace.set_ylabel("Conv_a")
        ax_trace.set_ylim(0, 2)
        ax_trace.grid(True, linestyle='--', alpha=0.5)
        ax_trace.legend(loc='best', framealpha=0.85)

        ax_var.plot(fe_vals, var_vals, color="#7f0000", linewidth=2.0, label="variance")
        ax_var.fill_between(fe_vals, 0, var_vals, color="#b22222", alpha=0.20)
        ax_var.set_xlabel("Function Evaluations (FEs)")
        ax_var.set_ylabel("Var")
        ax_var.grid(True, linestyle='--', alpha=0.5)
        ax_var.legend(loc='best', framealpha=0.85)

        save_path = os.path.join(output_dir, f"F{f_num}_conv_a_stats.png")
        fig.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ F{f_num} 的 Conv_a 均值/方差图已保存至 -> {save_path}")


def generate_all_plots(task_md5):
    extract_csv(task_md5)
    plot_highlight_functions(task_md5)
    plot_conv_a_traces(task_md5)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python plot_final_battle.py <task_md5>")

    generate_all_plots(sys.argv[1])
