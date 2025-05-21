import pandas as pd
import numpy as np

# 配置参数
input_file = "核桃仁表型信息_重新标定_0.02寻参.xlsx"  # 包含预测结果的Excel文件
sheet_name = "Sheet1"  # 工作表名称
actual_col = "g"  # 实际值列名（需根据实际情况修改）

# 读取数据
df = pd.read_excel(input_file, sheet_name=sheet_name)

# 准备数据
actual_values = df[actual_col].values
pred_columns = [col for col in df.columns if col.startswith("Result_")]

# 初始化最优参数
min_error = float('inf')
best_h = None
best_p = None

# 遍历所有预测列
for pred_col in pred_columns:
    try:
        # 从列名解析y和z参数
        params = pred_col.split("_")
        y = float(params[1].split("=")[1])
        z = float(params[2].split("=")[1])

        # 计算MAE（平均绝对误差）
        pred_values = df[pred_col].values
        error = np.mean(np.abs(pred_values - actual_values))

        # 更新最优参数
        if error < min_error:
            min_error = error
            best_h = y
            best_p = z
    except Exception as e:
        print(f"跳过无效列 {pred_col}: {str(e)}")

# 输出结果
print(f"\n最优参数组合：")
print(f"y = {best_h:.2f}")
print(f"z = {best_p:.2f}")
print(f"最小平均绝对误差：{min_error:.4f}")

# 保存结果到新文件（可选）
result_df = pd.DataFrame({
    "最佳参数": [f"h={best_h:.2f}, p={best_p:.2f}"],
    "最小误差": [min_error]
})
result_df.to_excel("best_parameters.xlsx", index=False)