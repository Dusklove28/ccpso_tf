import os
import pickle
import sys
import pandas as pd

# 确保能找到项目中的类
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 必须提前引入，否则 pickle 无法反序列化
try:
    from matAgent.pso import PsoSwarm
    from matAgent.clpso import ClpsoSwarm
    from matAgent.testpso import TestpsoSwarm
    from matAgent.rlepso import RlepsoSwarm
    from matAgent.ccpso_50d import FiftyDimCCPsoSwarm
    from matAgent.rl_ccpso_eval import RlCCPsoSwarm
except Exception:
    pass

def generate_excel():
    target_md5 = "e14070c2f46a4a1489ceaf9d3564ed75" # 你的任务哈希
    task_dir = os.path.join("data", "task", target_md5)
    pickle_path = os.path.join(task_dir, "result.pickle")

    if not os.path.exists(pickle_path):
        print(f"❌ 还是没找到结果文件！")
        print(f"请检查这个路径是否存在: {os.path.abspath(pickle_path)}")
        print("\n如果路径不存在，说明主进程还没计算完。请再等 1 分钟。")
        return

    print(f"🚀 正在解析最终大决战结果: {target_md5}")
    
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
            
        # 提取汇总数据
        # 结构：data['result'][0] 包含了 type, average_ranks, functions, result
        summary = data['result'][0]
        detailed_fits = summary['result'] # 字典: {f_num: {opt_name: result_matrix}}
        avg_ranks = summary['average_ranks']

        # 1. 构造基础表格
        rows = []
        for f_num in sorted(detailed_fits.keys()):
            row = {'Function': f'F{f_num}'}
            for opt_name, res_data in detailed_fits[f_num].items():
                # 获取最后一次运行的最终适应度 (result 矩阵的最后一行第3列)
                # 原代码中 opt_data['result'] 是平均后的 trace，取最后一行
                final_val = res_data['result'][-1][2]
                # 将对象转为类名字符串
                name_str = getattr(opt_name, '__name__', str(opt_name))
                row[name_str] = final_val
            rows.append(row)
        
        df = pd.DataFrame(rows)

        # 2. 插入平均排名行
        rank_row = {'Function': '--- Average Rank ---'}
        for opt_name, rank_val in avg_ranks.items():
            name_str = getattr(opt_name, '__name__', str(opt_name))
            rank_row[name_str] = round(rank_val, 4)
        
        df = pd.concat([df, pd.DataFrame([rank_row])], ignore_index=True)

        # 3. 保存
        output_name = "CCPso_VS_Others_Final.xlsx"
        df.to_excel(output_name, index=False)
        print(f"\n🎉 恭喜！你的学术成果已保存至: {output_name}")
        print("-" * 50)
        print("最终平均排名（分值越低越牛）：")
        for name, rank in sorted(avg_ranks.items(), key=lambda x: x[1]):
            print(f"  {getattr(name, '__name__', name)}: {rank:.4f}")
        print("-" * 50)

    except Exception as e:
        print(f"❌ 解析过程发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_excel()