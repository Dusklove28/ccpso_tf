import os
import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 解决 Linux/WSL 下中文字体显示为方块的问题
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class MockClass: pass


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return type(name, (MockClass,), {"__module__": module})


def extract_csv():
    path = "data/task/43fd48ae8450de74e7eacd0967b0ec19/result.pickle"
    if not os.path.exists(path):
        return

    with open(path, "rb") as f:
        obj = pickle.load(f)

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


def plot_highlight_functions():
    target_md5 = "43fd48ae8450de74e7eacd0967b0ec19"
    task_dir = os.path.join("data", "task", target_md5)
    pickle_path = os.path.join(task_dir, "result.pickle")
    output_dir = "final_battle_plots"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(pickle_path):
        print(f"❌ 找不到结果文件：{pickle_path}")
        return

    print("🚀 正在绘制学术级对比图...")
    with open(pickle_path, 'rb') as f:
        data = SafeUnpickler(f).load()

    real_results = data['result'][0]['result']
    target_functions = list(range(1, 29))

    # 顶会论文经典配色 (Colorblind-friendly)
    # 精确映射你真实的算法名称
    LABEL_MAP = {
        "PSOorigin": "PSO",
        "PSOtrain": "RLPSO",
        "Conv_PSOtrain": "RL_CCPSO"
    }

    colors = {
        'RL_CCPSO': '#e41a1c',  # 鲜艳的红色，突出你的算法
        'RLPSO': '#377eb8',  # 沉稳的蓝色
        'PSO': '#4daf4a'  # 经典的绿色
    }

    markers = {
        'RL_CCPSO': 'o',
        'RLPSO': 's',
        'PSO': '^'
    }

    for f_num in target_functions:
        if f_num in real_results:
            opt_dicts = real_results[f_num]
        elif str(f_num) in real_results:
            opt_dicts = real_results[str(f_num)]
        else:
            continue

        plt.figure(figsize=(10, 6))

        # --- 计算 symlog 阈值，防止 Y 轴刻度问题 ---
        all_y = []
        for opt_name, res_data in opt_dicts.items():
            matrix = res_data['result']
            # 修复 NumPy 真值判断 Bug
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

        # --- 开始画图 ---
        for opt_name, res_data in opt_dicts.items():
            matrix = res_data['result']
            # 修复 NumPy 真值判断 Bug
            if matrix is None or len(matrix) == 0:
                continue

            # 使用映射获取学术展示名称
            label_name = LABEL_MAP.get(opt_name, str(opt_name))

            x_vals = [row[0] for row in matrix]
            y_vals = [row[2] if len(row) > 2 else row[-1] for row in matrix]

            color = colors.get(label_name, '#999999')
            marker = markers.get(label_name, 'x')
            mark_step = max(1, len(x_vals) // 15)

            # 强调 Ours
            lw = 2.5 if 'Ours' in label_name else 1.5
            zo = 10 if 'Ours' in label_name else 5

            plt.plot(x_vals, y_vals, label=label_name, color=color, marker=marker,
                     markevery=mark_step, linewidth=lw, zorder=zo, alpha=0.9)

        plt.title(f"Convergence Curves on 30D Complex Function F{f_num}", fontsize=15, fontweight='bold')
        plt.xlabel("Function Evaluations (FEs)", fontsize=13)
        plt.ylabel("Fitness Value (Log Scale)", fontsize=13)

        # 启用 symlog 坐标轴并强制划分刻度
        ax = plt.gca()
        ax.set_yscale('symlog', linthresh=linthresh)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))

        def y_formatter(val, pos):
            if val == 0:
                return "0"
            elif abs(val) < 1e-3 or abs(val) >= 1e4:
                return f"{val:.1e}"
            else:
                return f"{val:.4g}"

        ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_formatter))

        plt.legend(fontsize=11, loc='best', framealpha=0.8)
        plt.grid(True, linestyle='--', alpha=0.5)

        save_path = os.path.join(output_dir, f"F{f_num}_convergence_academic.png")
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close()

        print(f"✅ F{f_num} 的学术高清红蓝对决图已保存至 -> {save_path}")


if __name__ == "__main__":
    extract_csv()
    plot_highlight_functions()