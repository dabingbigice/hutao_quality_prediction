import pandas as pd
import numpy as np

# 读取Excel数据
df = pd.read_excel('核桃仁表型信息_重新标定_h_0.84.xlsx')

# 计算离心率（需确保a > b）
# df['e'] = np.sqrt(df['a']**2 - df['b']**2) / df['a']

# 处理异常值（例如a=0或b>a的情况）
# df['e'] = df['e'].apply(lambda x: x if not np.isnan(x) else 0)  # 无效值设为0
# df['hutao_da'] = (df['hutao_a'] + df['hutao_b'] + 0.58) / 3

hutao_a = df['hutao_a']
hutao_b = df['hutao_b']
hutao_area = df['hutao_area']
# 核桃仁h
h = 0.84

# abh算术平均值
arithmetic_a_b_h_avg = (hutao_a + hutao_b + h) / 3
df['arithmetic_a_b_h_avg']=arithmetic_a_b_h_avg
# abh几何平均值
geometry_a_b_h_avg = (hutao_a * hutao_b * h) ** (1 / 3)

df['geometry_a_b_h_avg']=geometry_a_b_h_avg
# 形状索引
hutao_SI = 2 * hutao_a / (h + hutao_b)
df['hutao_SI']=hutao_SI
# 厚度方向的伸长
hutao_ET = hutao_a / h
df['hutao_ET']=hutao_ET

# 垂直方向的伸长
hutao_EV = hutao_b / h
df['hutao_EV']=hutao_EV

# 球形度
fai = (geometry_a_b_h_avg / hutao_a) * 100
df['fai']=fai


# df['fai'] = (df['hutao_dg'] / df['hutao_a']) * 100
# df['hutao_c'] = np.sqrt(df['hutao_a'] ** 2 - df['hutao_b'] ** 2)
# df['arithmetic_a_b_avg'] = (df['hutao_a'] + df['hutao_b']) / 2
# df['geometry_a_b_avg'] = (df['hutao_a'] * df['hutao_b']) ** (1 / 2)

# 保存到新Excel文件
df.to_excel('核桃仁表型信息_重新标定_h_0.84.xlsx', index=False)
