import tensorflow as tf

print("TensorFlow 版本:", tf.__version__)
print("GPU 设备数量:", len(tf.config.list_physical_devices('GPU')))

try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("可用 GPU:")
        for gpu in gpus:
            print(f"  - {gpu}")

        # 显示 GPU 详细信息
        print("\nGPU 详细信息:")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  {gpu}: 内存增长已启用")
    else:
        print("未检测到 GPU，将使用 CPU 运行")
except Exception as e:
    print(f"GPU 检测出错：{e}")
