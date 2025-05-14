import cv2
import numpy as np
import math
from datetime import datetime

# 记录开始时间
start_time = datetime.now()

# 读取语义分割掩码
mask = cv2.imread("hutao.png", cv2.IMREAD_GRAYSCALE)
original_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# 计算原始面积
original_area = cv2.countNonZero(mask)

# 查找轮廓
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if not contours:
    print("未检测到有效轮廓")
    exit()

# 取最大轮廓拟合椭圆
max_contour = max(contours, key=cv2.contourArea)
ellipse = cv2.fitEllipse(max_contour)
a_major, a_minor = ellipse[1][0] / 2, ellipse[1][1] / 2

# 计算椭圆面积
ellipse_area = math.pi * a_major * a_minor
area_diff = abs(original_area - ellipse_area)

# 绘制结果
result_image = original_image.copy()
cv2.drawContours(result_image, [max_contour], -1, (0, 255, 0), 1)
cv2.ellipse(result_image, ellipse, (0, 0, 255), 2)


def calculate_errors(mask, ellipse):
    errors = []
    center = (ellipse[0][0], ellipse[0][1])
    axes = (ellipse[1][0] / 2, ellipse[1][1] / 2)
    angle = math.radians(ellipse[2])

    # 生成椭圆掩码
    ellipse_mask = np.zeros_like(mask)
    cv2.ellipse(ellipse_mask, ellipse, 255, -1)

    # 计算IoU
    intersection = cv2.bitwise_and(mask, ellipse_mask)
    union = cv2.bitwise_or(mask, ellipse_mask)
    iou = np.sum(intersection) / np.sum(union)

    # 遍历所有轮廓点计算误差
    for point in max_contour:
        x, y = point[0]

        # 坐标变换到椭圆坐标系
        dx = x - center[0]
        dy = y - center[1]
        x_rot = dx * math.cos(angle) + dy * math.sin(angle)
        y_rot = -dx * math.sin(angle) + dy * math.cos(angle)

        # 代数距离误差
        algebraic_error = (x_rot ** 2) / (axes[0] ** 2) + (y_rot ** 2) / (axes[1] ** 2) - 1
        errors.append(abs(algebraic_error))

    return iou, errors


# 计算误差指标
iou, errors = calculate_errors(mask, ellipse)
error_percent = [e * 100 for e in errors]  # 转换为百分比

# 统计指标
stats = {
    'avg_error': np.mean(error_percent),
    'max_error': np.max(error_percent),
    'min_error': np.min(error_percent),
    'std_error': np.std(error_percent),
    'area_diff': area_diff,
    'proc_time': (datetime.now() - start_time).total_seconds()
}

# 绘制文字信息
text_lines = [
    f"IoU: {iou:.3f}",
    f"AvgErr: {stats['avg_error']:.1f}%",
    f"Min/MaxErr: {stats['min_error']:.1f}%/{stats['max_error']:.1f}%",
    f"StdErr: {stats['std_error']:.1f}%",
    f"AreaDiff: {stats['area_diff']:.1f} px²",
    f"Time: {stats['proc_time']:.1f}s"
]

# 逐行添加文字
y_start = 30
line_spacing = 25
for i, line in enumerate(text_lines):
    y = y_start + i * line_spacing
    cv2.putText(result_image, line, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

# 保存并输出结果
cv2.imwrite("fitted_ellipse.png", result_image)
print(f"椭圆参数：长轴 {2 * a_major:.1f}px，短轴 {2 * a_minor:.1f}px")
print("误差统计：")
print(f"  - 平均误差: {stats['avg_error']:.1f}%")
print(f"  - 误差范围: {stats['min_error']:.1f}%~{stats['max_error']:.1f}%")
print(f"  - 误差标准差: {stats['std_error']:.1f}%")
print(f"面积差异: {stats['area_diff']:.1f} px²")
print(f"处理耗时: {stats['proc_time']:.1f}s")