import os
import pickle
import sys
import matplotlib.pyplot as plt
import platform
import pathlib

# 跨平台路径兼容补丁
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from matAgent.pso import PsoSwarm
    from matAgent.clpso import ClpsoSwarm
    from matAgent.testpso import TestpsoSwarm
    from matAgent.rlepso import RlepsoSwarm
    from matAgent.ccpso import FiftyDimCCPsoSwarm
    from matAgent.rl_ccpso_eval import RlCCPsoSwarm
except Exception:
    pass


def replot_graphs():
    # 极其重要：请把这里改成你这次 56 小时任务的 MD5 文件夹名！
    target_md5 = "e14070c2f46a4a1489ceaf9d3564ed75"

    task_dir = os.path.join("data", "task", target_md5)
    pickle_path = os.path.join(task_dir, "result.pickle")

    # 新建一个专门存放 RL 对比图的文件夹
    img_dir = os.path.join("data", "img", "RL_Only_Compare")
    os.makedirs(img_dir, exist_ok=True)

    if not os.path.exists(pickle_path):
        print(f"找不到结果文件：{pickle_path}")
        return

    print("正在读取底层数据，准备重新绘制纯净对比图...")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    summary = data['result'][0]
    detailed_fits = summary['result']

    # 遍历 28 个函数
    for f_num in sorted(detailed_fits.keys()):
        plt.figure(figsize=(10, 6))

        # 只提取特定的两种算法数据
        for opt_name, res_data in detailed_fits[f_num].items():
            name_str = getattr(opt_name, '__name__', str(opt_name))

            # 统一转换为大写，彻底消除大小写匹配错误
            name_upper = name_str.upper()

            if 'RLEPSO' in name_upper or 'CCPSO' in name_upper:
                trace = res_data['result']  # 取出收敛轨迹
                fes = [step[0] for step in trace]
                fits = [step[2] for step in trace]

                # 绘制折线图，设置线宽
                plt.plot(fes, fits, label=name_str, linewidth=2.5)

        # 设置图表格式
        plt.title(f'Convergence Curve - CEC2013 F{f_num} (RL Ablation)')
        plt.xlabel('Function Evaluations (FEs)')
        plt.ylabel('Fitness (Log Scale)')
        plt.yscale('symlog') # 使用对称对数轴 (Symmetrical Log)，完美支持负数显示
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)

        # 保存图片
        save_path = os.path.join(img_dir, f'F{f_num}_RL_Compare.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"函数 F{f_num} 的全新对比图已生成: {save_path}")

    print(f"\n完美！所有 28 张图片已存放至 {img_dir} 目录中！")


if __name__ == "__main__":
    replot_graphs()