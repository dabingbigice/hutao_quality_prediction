import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# 配置参数
input_file = "核桃仁表型信息_调试_0.02.xlsx"
sheet_name = "Sheet1"
actual_col = "g"
pred_col = "Result_y=0.78_z=0.58"
output_image = "scatter_plot_0.02.png"

# 读取数据
df = pd.read_excel(input_file, sheet_name=sheet_name)
actual = df[actual_col].values
pred = df[pred_col].values

# 计算指标
r2 = r2_score(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))
abs_errors = np.abs(actual - pred)
max_error = np.max(abs_errors)
max_idx = np.argmax(abs_errors)  # 获取最大误差点的索引

# 创建画布
plt.figure(figsize=(10, 8), dpi=120)

# 绘制散点图
sc = plt.scatter(actual, pred, c=abs_errors, cmap='viridis', alpha=0.7, edgecolors='w')
plt.colorbar(sc, label='Absolute Error')

# 绘制对角线
min_val = min(actual.min(), pred.min())
max_val = max(actual.max(), pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1)

# 标注最大误差点
plt.annotate(f'Max Error: {max_error:.2f}',
             xy=(actual[max_idx], pred[max_idx]),
             xytext=(actual[max_idx]+0.1, pred[max_idx]-0.1),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10,
             color='red')

# 添加指标文本
textstr = '\n'.join((
    fr'$R^2 = {r2:.3f}$',
    fr'RMSE = {rmse:.3f}',
    fr'Max Error = {max_error:.3f}'))
plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
               fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

# 图表装饰
plt.title(f'Actual vs Predicted Values\n({pred_col})', fontsize=14)
plt.xlabel('Actual Values (g)', fontsize=12)
plt.ylabel('Predicted Values', fontsize=12)
plt.grid(alpha=0.2)

# 保存输出
plt.tight_layout()
plt.savefig(output_image, bbox_inches='tight')
print(f"图表已保存至 {output_image}")