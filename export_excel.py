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
    # 已经为你替换好了最新的 50 维测试任务 MD5
    target_md5 = "b598bae2ebc31d45b33b7bde3a3cfb1c" 
    task_dir = os.path.join("data", "task", target_md5)
    pickle_path = os.path.join(task_dir, "result.pickle")

    if not os.path.exists(pickle_path):
        print(f"❌ 还是没找到结果文件！")
        print(f"请检查这个路径是否存在: {os.path.abspath(pickle_path)}")
        return

    print(f"🚀 正在解析最终 50 维大决战结果: {target_md5}")
    
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
            
        # 【核心修复】：绕过不存在的 0 号索引，直接获取字典
        detailed_fits = data['result'] 

        # 1. 构造基础表格
        rows = []
        for f_num in sorted(detailed_fits.keys()):
            row = {'Function': f'F{f_num}'}
            for opt_name, res_data in detailed_fits[f_num].items():
                # 取最后一次运行的最终适应度 (result 矩阵的最后一行第3列)
                final_val = res_data['result'][-1][2]
                
                # 清洗类名，让 Excel 表头好看点
                name_str = getattr(opt_name, '__name__', str(opt_name)).split('.')[-1].replace("'>", "").replace("train", "")
                row[name_str] = final_val
            rows.append(row)
        
        df = pd.DataFrame(rows)

        # 2. 插入平均排名行（Pandas 手动计算，分值越小越好）
        opt_cols = [c for c in df.columns if c != 'Function']
        ranks = df[opt_cols].rank(axis=1, ascending=True)
        mean_ranks = ranks.mean()

        rank_row = {'Function': '--- Average Rank ---'}
        for col in opt_cols:
            rank_row[col] = round(mean_ranks[col], 4)
        
        df = pd.concat([df, pd.DataFrame([rank_row])], ignore_index=True)

        # 3. 保存
        output_name = "CCPso_VS_Others_Final.xlsx"
        df.to_excel(output_name, index=False)
        print(f"\n🎉 恭喜！你的学术成果已保存至: {output_name}")
        print("-" * 50)
        print("🏆 最终平均排名（分值越低越牛）：")
        for col in sorted(opt_cols, key=lambda c: mean_ranks[c]):
            print(f"  {col}: {mean_ranks[col]:.4f}")
        print("-" * 50)

    except Exception as e:
        print(f"❌ 解析过程发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_excel()