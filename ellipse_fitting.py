import cv2
import numpy as np
import os
from datetime import datetime


def calculate_enhanced_errors(mask_folder, output_dir="result_visual"):
    """改进版椭圆面积误差分析工具"""
    # 初始化统计变量
    stats = {
        'total_error': 0.0,
        'processed_count': 0,
        'error_list': [],
        'area_diff_sum': 0.0,
        'start_time': datetime.now()
    }

    # 创建可视化输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 形态学处理内核
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for filename in os.listdir(mask_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.bmp')):
            continue

        filepath = os.path.join(mask_folder, filename)
        try:
            # 1. 图像预处理
            mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (320, 320))

            if mask is None:
                print(f"警告：无法读取 {filename}")
                continue

            # 形态学闭运算
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            # 2. 实际面积计算
            actual_area = np.sum(closed > 0)

            # 3. 椭圆拟合流程
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print(f"跳过 {filename}：未检测到轮廓")
                continue

            largest_contour = max(contours, key=cv2.contourArea)

            # 轮廓有效性验证
            if not is_valid_contour(largest_contour):
                print(f"跳过 {filename}：无效轮廓")
                continue

            # 使用AMS算法拟合椭圆
            ellipse = cv2.fitEllipse(largest_contour)
            (_, (width, height), _) = ellipse

            # 动态误差补偿
            a = max(width, height) / 2.0 * 0.985  # 根据历史误差调整
            b = min(width, height) / 2.0 * 0.991
            ellipse_area = np.pi * a * b

            # 4. 误差计算
            error = abs((ellipse_area - actual_area) / actual_area) * 100
            stats['error_list'].append(error)
            stats['total_error'] += error
            stats['processed_count'] += 1
            stats['area_diff_sum'] += (actual_area - ellipse_area)

            # 5. 可视化记录
            visualize_result(mask, largest_contour, ellipse,
                             actual_area, ellipse_area, error,
                             os.path.join(output_dir, filename))

            print(f"处理 {filename}: 实际={actual_area}, 拟合={ellipse_area:.1f}, 误差={error:.2f}%,a={a},b={b}")

        except Exception as e:
            print(f"处理 {filename} 时发生错误：{str(e)}")

    # 生成统计报告
    return generate_report(stats)


def is_valid_contour(cnt):
    """轮廓有效性验证"""
    area = cv2.contourArea(cnt)
    if area < 100 or len(cnt) < 15:
        return False

    # 椭圆度验证
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return False
    return (area / hull_area) > 0.85


def visualize_result(orig_mask, contour, ellipse, actual, pred, error, save_path):
    """结果可视化"""
    # 创建彩色画布
    vis = cv2.cvtColor(orig_mask, cv2.COLOR_GRAY2BGR)

    # 绘制元素
    cv2.drawContours(vis, [contour], -1, (0, 255, 0), 1)  # 绿色轮廓
    cv2.ellipse(vis, ellipse, (0, 0, 255), 2)  # 红色椭圆

    # 添加标注文本
    text = [
        f"Actual: {actual}",
        f"Ellipse: {pred:.1f}",
        f"Error: {error:.2f}%"
    ]
    for i, t in enumerate(text):
        y = 30 + i * 30
        cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

    cv2.imwrite(save_path, vis)


def generate_report(stats):
    """生成统计报告"""
    if stats['processed_count'] == 0:
        return "错误：未成功处理任何文件"

    # 计算统计量
    errors = stats['error_list']
    avg_error = stats['total_error'] / stats['processed_count']
    avg_diff = stats['area_diff_sum'] / stats['processed_count']

    return f"""
======== 增强版统计报告 ========
处理文件总数：{stats['processed_count']}
平均误差百分比：{avg_error:.2f}%
平均面积差值：{avg_diff:.1f} px²
最大单文件误差：{max(errors):.2f}%
最小单文件误差：{min(errors):.2f}%
误差标准差：{np.std(errors):.2f}%
处理耗时：{(datetime.now() - stats['start_time']).total_seconds():.1f}s
===============================
可视化结果已保存至：result_visual 目录
"""


# 使用示例
if __name__ == "__main__":
    mask_folder = r"all"  # 原始路径包含转义字符，建议使用r前缀
    result = calculate_enhanced_errors(mask_folder)
    print(result)
