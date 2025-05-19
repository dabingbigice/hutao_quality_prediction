import pandas as pd
import numpy as np

# 读取Excel数据
df = pd.read_excel('核桃仁表型信息_重新标定.xlsx')

# 计算离心率（需确保a > b）
# df['e'] = np.sqrt(df['a']**2 - df['b']**2) / df['a']

# 处理异常值（例如a=0或b>a的情况）
# df['e'] = df['e'].apply(lambda x: x if not np.isnan(x) else 0)  # 无效值设为0
# df['hutao_da'] = (df['hutao_a'] + df['hutao_b'] + 0.58) / 3

# df['hutao_dg'] = (df['hutao_a'] * df['hutao_b'] * 0.58) ** (1 / 3)

# df['fai'] = (df['hutao_dg'] / df['hutao_a']) * 100
df['hutao_c'] = np.sqrt(df['hutao_a'] ** 2 - df['hutao_b'] ** 2)

# 保存到新Excel文件
df.to_excel('核桃仁表型信息_重新标定.xlsx', index=False)
