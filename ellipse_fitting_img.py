import cv2
import numpy as np
import os
from datetime import datetime


def single_file_predict(image_path, output_dir="result_visual"):
    """单文件预测函数"""
    # 创建可视化输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 初始化结果字典
    result = {
        'status': 'success',
        'message': '',
        'actual_area': 0,
        'predicted_area': 0,
        'error_percent': 0,
        'axes': (0, 0),
        'vis_path': ''
    }

    try:
        # 1. 读取并预处理图像
        mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError("无法读取图像文件")

        # 形态学闭运算
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 2. 实际面积计算
        actual_area = np.sum(closed > 0)
        result['actual_area'] = actual_area

        # 3. 椭圆拟合流程
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("未检测到有效轮廓")

        largest_contour = max(contours, key=cv2.contourArea)

        # 轮廓有效性验证
        if not is_valid_contour(largest_contour):
            raise ValueError("无效轮廓")

        # 使用算法拟合椭圆
        ellipse = cv2.fitEllipse(largest_contour)
        (_, (width, height), _) = ellipse

        # 动态误差补偿
        a = max(width, height) / 2.0 * 0.985
        b = min(width, height) / 2.0 * 0.991
        ellipse_area = np.pi * a * b

        # 记录预测结果
        result['predicted_area'] = ellipse_area
        result['error_percent'] = abs((ellipse_area - actual_area) / actual_area) * 100
        result['axes'] = (a * 2, b * 2)  # 返回完整轴长
        error = abs((ellipse_area - actual_area) / actual_area) * 100

        print(f"处理 {image_path}: 实际={actual_area}, 拟合={ellipse_area:.1f}, 误差={error:.2f}%,a={a},b={b}")

        # 4. 生成可视化结果
        base_name = os.path.basename(image_path)
        vis_path = os.path.join(output_dir, f"result_{base_name}")
        visualize_result(mask, largest_contour, ellipse,
                         actual_area, ellipse_area,
                         result['error_percent'], vis_path)
        result['vis_path'] = vis_path

    except Exception as e:
        result['status'] = 'error'
        result['message'] = str(e)

    return result, a, b


def is_valid_contour(cnt):
    """轮廓有效性验证（与原函数相同）"""
    area = cv2.contourArea(cnt)
    if area < 100 or len(cnt) < 15:
        return False
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return False
    return (area / hull_area) > 0.85


def visualize_result(orig_mask, contour, ellipse, actual, pred, error, save_path):
    """可视化函数（与原函数相同）"""
    vis = cv2.cvtColor(orig_mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, [contour], -1, (0, 255, 0), 1)
    cv2.ellipse(vis, ellipse, (0, 0, 255), 2)
    text = [
        f"Actual: {actual}",
        f"Predicted: {pred:.1f}",
        f"Error: {error:.2f}%"
    ]
    for i, t in enumerate(text):
        y = 30 + i * 30
        cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(save_path, vis)


# 使用示例
if __name__ == "__main__":
    # 测试单文件预测
    test_file = "hutao.png"  # 修改为实际文件路径
    result = single_file_predict(test_file)

    # 打印结果
    if result['status'] == 'success':
        print("======== 预测结果 ========")
        print(f"实际面积: {result['actual_area']} px²")
        print(f"预测面积: {result['predicted_area']:.1f} px²")
        print(f"长轴长度: {result['axes'][0]:.1f} px")
        print(f"短轴长度: {result['axes'][1]:.1f} px")
        print(f"面积误差: {result['error_percent']:.2f}%")
        print(f"可视化结果: {result['vis_path']}")
    else:
        print(f"处理失败: {result['message']}")
