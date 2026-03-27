import os
import shutil


def auto_clean_evaluation_folders():
    task_dir = os.path.join("data", "task")
    clean_dir = os.path.join(task_dir, "clean")

    # 确保 clean 目录存在 (你已经建了，但加一句更保险)
    os.makedirs(clean_dir, exist_ok=True)

    if not os.path.exists(task_dir):
        print(f"❌ 找不到目录: {task_dir}")
        return

    print(f"🧹 准备开始清理！目标转移目录: {clean_dir}\n")

    moved_count = 0

    # 遍历所有 MD5 文件夹
    for md5_name in os.listdir(task_dir):
        # 排除 clean 目录自身，以及非文件夹的文件
        if md5_name == "clean":
            continue

        folder_path = os.path.join(task_dir, md5_name)

        if os.path.isdir(folder_path):
            has_h5 = False
            has_result = False

            # 深度扫描文件夹里的所有文件
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith('.h5'):
                        has_h5 = True
                    if file == 'result.pickle' or file == 'task.pickle':
                        has_result = True

            # 核心搬运逻辑：没有 .h5 模型，且包含测试结果/任务配置，确定是纯测试文件夹！
            if not has_h5 and has_result:
                dest_path = os.path.join(clean_dir, md5_name)
                try:
                    # 将整个 MD5 文件夹移动到 clean 目录下
                    shutil.move(folder_path, dest_path)
                    moved_count += 1
                    print(f"✅ 成功移走: 【 {md5_name} 】 -> clean目录")
                except Exception as e:
                    print(f"❌ 移动失败 {md5_name}: {e}")
            elif has_h5:
                print(f"🛡️ 保护跳过: {md5_name} (包含 .h5 极品模型，绝对安全)")

    print("\n==================================================")
    print(f"🎉 清理大扫除完成！共成功把 {moved_count} 个历史测试文件夹打入冷宫。")
    print("🚀 战场已彻底清空！现在你可以毫无顾虑地去运行 python main.py，开启 50 维的终极之战了！")


if __name__ == "__main__":
    auto_clean_evaluation_folders()