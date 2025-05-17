import pandas as pd
import numpy as np

# 读取Excel数据
df = pd.read_excel('核桃仁表型信息.xlsx')

# 计算离心率（需确保a > b）
# df['e'] = np.sqrt(df['a']**2 - df['b']**2) / df['a']

# 处理异常值（例如a=0或b>a的情况）
# df['e'] = df['e'].apply(lambda x: x if not np.isnan(x) else 0)  # 无效值设为0
df['hutao_a'] = df['a'] * 0.02596
df['hutao_b'] = df['b'] * 0.02596
df['hutao_a/b'] = df['hutao_a'] / df['hutao_b']
# 保存到新Excel文件
df.to_excel('核桃仁表型信息.xlsx', index=False)
