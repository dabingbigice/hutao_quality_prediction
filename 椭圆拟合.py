import cv2
import numpy as np
import math

# 读取语义分割掩码（确保是二值图）
mask = cv2.imread("hutao.png", cv2.IMREAD_GRAYSCALE)
original_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 用于可视化的彩色图像

# 查找轮廓
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if not contours:
    print("未检测到有效轮廓")
    exit()

# 取最大轮廓拟合椭圆
max_contour = max(contours, key=cv2.contourArea)
ellipse = cv2.fitEllipseAMS(max_contour)
(a_major, a_minor) = (ellipse[1][0] / 2, ellipse[1][1] / 2)  # 长短半轴

# 在原始图像上绘制结果
result_image = original_image.copy()
cv2.drawContours(result_image, [max_contour], -1, (0, 255, 0), 1)  # 绿色绘制原始轮廓
cv2.ellipse(result_image, ellipse, (0, 0, 255), 2)  # 红色绘制拟合椭圆


# 计算拟合误差
def calculate_fitting_error(mask, ellipse):
    # 生成椭圆掩码
    ellipse_mask = np.zeros_like(mask)
    cv2.ellipse(ellipse_mask, ellipse, 255, -1)  # 填充椭圆

    # 计算IoU
    intersection = cv2.bitwise_and(mask, ellipse_mask)
    union = cv2.bitwise_or(mask, ellipse_mask)
    iou = np.sum(intersection) / np.sum(union)

    # 计算轮廓点与椭圆边界的平均距离误差
    total_distance = 0
    for point in max_contour:
        x, y = point[0]
        center = (int(ellipse[0][0]), int(ellipse[0][1]))
        axes = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
        angle = ellipse[2]

        # 计算点到椭圆的代数距离（近似）
        theta = -math.radians(angle)
        dx = x - center[0]
        dy = y - center[1]
        rotated_x = dx * math.cos(theta) - dy * math.sin(theta)
        rotated_y = dx * math.sin(theta) + dy * math.cos(theta)
        distance = (rotated_x ** 2) / (axes[0] ** 2) + (rotated_y ** 2) / (axes[1] ** 2) - 1
        total_distance += abs(distance)

    avg_distance = total_distance / len(max_contour)
    return iou, avg_distance


iou, avg_dist = calculate_fitting_error(mask, ellipse)

# 添加文字标注
text = f"IoU: {iou:.3f}  AvgDist: {avg_dist:.2f}"
cv2.putText(result_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# 保存结果
cv2.imwrite("fitted_ellipse.png", result_image)
print(f"拟合结果已保存为 fitted_ellipse.png")
print(f"长轴：{2 * a_major:.2f}px，短轴：{2 * a_minor:.2f}px")
print(f"交并比(IoU)：{iou:.4f}")
print(f"平均代数距离误差：{avg_dist:.4f}")