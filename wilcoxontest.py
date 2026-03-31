import pandas as pd
from scipy.stats import wilcoxon

# 读取数据
df = pd.read_excel('CCPso_VS_Others_Final.xlsx', sheet_name='Sheet1')
# 提取最后一行（平均排名行）之前的数据
df_functions = df[df['Function'].str.contains('F', na=False)].copy()
# 提取两列数值（转为 float）
rlepso = df_functions['RLEPSO'].astype(float)
ccpso = df_functions['RL_CCPSO50D'].astype(float)

# 执行 Wilcoxon 符号秩检验（单边：ccpso < rlepso）
stat, p_value = wilcoxon(r, rlepso, alternative='less')

print(f"Wilcoxon W 统计量（负秩和）: {stat}")
print(f"单边 p 值 (CCPSO < RLEPSO): {p_value:.6f}")

if p_value < 0.05:
    print("结论：RL_CCPSO50D 显著优于 RLEPSO (p < 0.05)")
else:
    print("结论：未检测到统计显著差异 (p >= 0.05)")