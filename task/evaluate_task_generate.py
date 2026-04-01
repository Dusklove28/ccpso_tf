import os
import re
from pathlib import Path

try:
    from matAgent.testpso import TestpsoSwarm
    from matAgent.ccpso import FiftyDimCCPsoSwarm
    from matAgent.rlepso import RlepsoSwarm
    from matAgent.rl_ccpso_eval import RlCCPsoSwarm
    from matAgent.pso import PsoSwarm
    from matAgent.clpso import ClpsoSwarm
except Exception:
    pass

def build_model_dict_from_log():
    log_file = "experiment.log"
    task_dir = os.path.join("data", "task")
    fun_model_rlepso = {}
    fun_model_ccpso = {}
    
    if not os.path.exists(log_file):
        print(f"❌ 找不到日志文件 {log_file}！")
        return {}, {}
        
    latest_md5_map = {}
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if 'run task' in line and 'single_train' in line:
                md5_match = re.search(r'run task ([a-fA-F0-9]{32})', line)
                if not md5_match: continue
                md5 = md5_match.group(1)
                
                fun_match = re.search(r"(?:'fun_nums'|'evaluate_function'):\s*\[(\d+)\]", line)
                if not fun_match: continue
                fun_num = int(fun_match.group(1))
                
                opt_type = None
                if 'FiftyDimCCPsoSwarm' in line or 'CCPSO_50D' in line or 'ccpso_50d' in line:
                    opt_type = 'CCPSO'
                elif 'TestpsoSwarm' in line or 'TESTPSO' in line or 'testpso' in line:
                    opt_type = 'RLEPSO'
                    
                if opt_type and fun_num:
                    latest_md5_map[(opt_type, fun_num)] = md5

    for (opt_type, fun_num), md5 in latest_md5_map.items():
        folder_path = os.path.join(task_dir, md5)
        # 🔥 报警器：如果文件夹不在 data/task/ 下，直接打印警告！
        if not os.path.isdir(folder_path): 
            print(f"⚠️ 警告：日志说有 {opt_type} 的 F{fun_num} (MD5: {md5})，但在 data/task/ 目录下没找到这个文件夹！")
            continue
        
        h5_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
        actor_files = [f for f in h5_files if 'actor' in f.lower()]
        if actor_files: h5_files = actor_files
        if not h5_files: continue
        
        h5_file = sorted(h5_files, key=lambda x: (len(x), x))[-1]
        model_path = Path(folder_path) / h5_file
        
        if opt_type == 'CCPSO':
            fun_model_ccpso.setdefault(fun_num, []).append(model_path)
        elif opt_type == 'RLEPSO':
            fun_model_rlepso.setdefault(fun_num, []).append(model_path)
            
    print(f"\n✅ 成功从硬盘找回 {len(fun_model_rlepso)} 个 RLEPSO 模型，{len(fun_model_ccpso)} 个 CCPSO 模型！")
    return fun_model_rlepso, fun_model_ccpso

def generate_evaluate_tasks():
    fun_model_rlepso, fun_model_ccpso = build_model_dict_from_log()
    
    optimizer_model_list = []
    
    if fun_model_rlepso:
        optimizer_model_list.append({
            'optimizer': RlepsoSwarm,
            'fun_model': fun_model_rlepso,
        })
    else:
        print("❌ 严重警告：没有找到任何 RLEPSO 模型！")
        
    if fun_model_ccpso:
        optimizer_model_list.append({
            'optimizer': RlCCPsoSwarm,
            'fun_model': fun_model_ccpso,
        })
    else:
        print("❌ 严重警告：没有找到任何 RL_CCPSO 模型！")
        
    if not fun_model_rlepso and not fun_model_ccpso:
        print("🚨 致命错误：RL模型全部丢失！请确保你把包含 .h5 的文件夹移动回了 data/task/ 目录！\n任务已终止，系统只会跑传统算法！")
        
    # 为传统算法伪造一个空模型字典
    dummy_fun_model = {i: [None] for i in range(1, 29)}
    
    optimizer_model_list.append({'optimizer': PsoSwarm, 'fun_model': dummy_fun_model})
    optimizer_model_list.append({'optimizer': ClpsoSwarm, 'fun_model': dummy_fun_model})
        
    group = 5
    max_fe = 10000
    n_part = 100
    real_eval_dim = 50   
    runtimes = 5
    
    new_result_evaluate_task_dic = {
        'type': 'new_result_evaluate',
        'optimizer_model_list': optimizer_model_list,
        'evaluate_function': list(range(1, 29, 1)),
        'group': group,
        'max_fe': max_fe,
        'n_part': n_part,
        'dim': real_eval_dim,
        'runtimes': runtimes,
    }
    
    return [new_result_evaluate_task_dic]

if __name__ == '__main__':
    generate_evaluate_tasks()