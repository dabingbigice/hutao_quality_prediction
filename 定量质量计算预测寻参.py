import pandas as pd
import numpy as np

# 配置参数
input_file = "核桃仁表型信息_重新标定.xlsx"   # 输入的Excel文件名
output_file = "核桃仁表型信息_重新标定_0.02寻参.xlsx" # 输出的Excel文件名
sheet_name = "Sheet1"       # 工作表名称
x_column = "hutao_area"              # x列的名称

# y和z的范围及步长
y_range = np.arange(0.6, 1.3 + 0.02, 0.02)  # 1.3包含在范围内
z_range = np.arange(0.5, 0.8 + 0.02, 0.02)  # 0.8包含在范围内

# 读取x数据
df = pd.read_excel(input_file, sheet_name=sheet_name)
x_values = df[x_column].values

# 遍历所有y和z组合
for y in y_range:
    y = round(y, 2)  # 确保精度
    for z in z_range:
        z = round(z, 2)
        # 计算新列
        column_name = f"Result_y={y}_z={z}"
        df[column_name] = x_values * y * z

# 保存结果到新文件
df.to_excel(output_file, index=False)
print(f"计算完成！结果已保存到 {output_file}")