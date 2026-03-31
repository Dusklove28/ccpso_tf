import os
import pickle
import matplotlib.pyplot as plt

# 解决 Linux/WSL 下中文字体显示为方块的问题
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class MockClass: pass


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return type(name, (MockClass,), {"__module__": module})


def plot_highlight_functions():
    target_md5 = "b598bae2ebc31d45b33b7bde3a3cfb1c"
    task_dir = os.path.join("data", "task", target_md5)
    pickle_path = os.path.join(task_dir, "result.pickle")
    output_dir = "final_battle_plots"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(pickle_path):
        print(f"❌ 找不到结果文件：{pickle_path}")
        return

    print("🚀 正在注入顶会级别调色盘...")
    with open(pickle_path, 'rb') as f:
        data = SafeUnpickler(f).load()

    real_results = data['result']

    # 你可以随时往这个列表里加别的函数编号 (比如 13, 16 等等)
    target_functions = list(range(1,29))

    # 顶会论文经典配色 (Colorblind-friendly)
    colors = {
        'RL_CCPSO (Ours)': '#e41a1c',  # 鲜艳的红色，突出你的算法
        'RLEPSO': '#377eb8',  # 沉稳的蓝色
        'CLPSO': '#ff7f00',  # 醒目的橙色
        'PSO': '#4daf4a'  # 经典的绿色
    }

    # 不同的标记符号，方便黑白打印时也能区分
    markers = {
        'RL_CCPSO (Ours)': 'o',
        'RLEPSO': 's',
        'CLPSO': 'D',
        'PSO': '^'
    }

    for f_num in target_functions:
        if f_num not in real_results: continue

        plt.figure(figsize=(10, 6))
        opt_dicts = real_results[f_num]

        for opt_name, res_data in opt_dicts.items():
            matrix = res_data['result']

            if matrix is None or len(matrix) == 0:
                continue

            # 【核心修复】：全部转小写，绝对无死角匹配
            name_raw = str(opt_name).lower()
            if 'ccpso' in name_raw:
                label_name = 'RL_CCPSO (Ours)'
            elif 'rlepso' in name_raw or 'testpso' in name_raw:
                label_name = 'RLEPSO'
            elif 'clpso' in name_raw:
                label_name = 'CLPSO'
            elif 'pso' in name_raw:
                label_name = 'PSO'
            else:
                label_name = str(opt_name).split('.')[-1].replace("'>", "")
            # else:
            #     continue

            x_vals = [row[0] for row in matrix]
            y_vals = [row[2] if len(row) > 2 else row[-1] for row in matrix]

            color = colors.get(label_name, '#999999')
            marker = markers.get(label_name, 'x')
            mark_step = max(1, len(x_vals) // 15)  # 控制标记点的密度，让图看起来更高级

            # 如果是你的算法，就把线条加粗 (linewidth=2.5)，并放在最上层 (zorder=10)
            lw = 2.5 if 'Ours' in label_name else 1.5
            zo = 10 if 'Ours' in label_name else 5

            plt.plot(x_vals, y_vals, label=label_name, color=color, marker=marker,
                     markevery=mark_step, linewidth=lw, zorder=zo, alpha=0.9)

        plt.title(f"Convergence Curves on 50D Complex Function F{f_num}", fontsize=15, fontweight='bold')
        plt.xlabel("Function Evaluations (FEs)", fontsize=13)
        plt.ylabel("Fitness Value (Log Scale)", fontsize=13)
        # plt.yscale('log')
        plt.yscale('symlog')
        plt.legend(fontsize=11, loc='best', framealpha=0.8)
        plt.grid(True, linestyle='--', alpha=0.5)

        save_path = os.path.join(output_dir, f"F{f_num}_convergence_academic.png")
        plt.savefig(save_path, dpi=400, bbox_inches='tight')  # 400 DPI，绝对高清无码
        plt.close()
        print(f"✅ F{f_num} 的学术高清红蓝对决图已保存至 -> {save_path}")


if __name__ == "__main__":
    plot_highlight_functions()