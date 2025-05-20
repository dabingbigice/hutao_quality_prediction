import cv2
import numpy as np
import os
from datetime import datetime
import math


def ellipse_perimeter(a, b):
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


def single_file_predict(image_path, area, output_dir="result_visual"):
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
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 2. 实际面积计算
        actual_area = area
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
        ellipse = cv2.fitEllipseDirect(largest_contour)
        (center, (width, height), angle) = ellipse

        ellipse_adj = (center,
                       (width * 0.92, height * 0.92),  # 轴长修正
                       angle)
        # 动态误差补偿
        a = max(width, height) / 2.0 * 0.92
        b = min(width, height) / 2.0 * 0.92
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
        visualize_result(mask, ellipse, ellipse_adj, actual_area, vis_path)
        result['vis_path'] = vis_path
        ellipse_perimeter(a, b)
        return result, a, b, ellipse_perimeter(a, b), error

    except Exception as e:
        result['status'] = 'error'
        result['message'] = str(e)
        print(e)
        return result, a, b, ellipse_perimeter(a, b), error


def is_center_region(x: float, y: float) -> bool:
    """
    判断坐标 (x, y) 是否位于图像中心区域（面积占整图的50%）
    ---
    参数:
      x: 横坐标 (0~319)
      y: 纵坐标 (0~319)
    返回:
      True: 在中心区域
      False: 不在中心区域
    """
    # 图像尺寸
    width, height = 320, 320

    # 计算中心区域边长（面积占50%时，边长为原尺寸的√0.5倍）
    center_size = int(width * math.sqrt(0.1))  # ≈ 226 像素
    margin = (width - center_size) // 2  # ≈ 47 像素

    # 计算中心区域边界
    x_min = margin
    x_max = width - margin - 1  # 坐标从0开始，需减1
    y_min = margin
    y_max = height - margin - 1

    # 判断坐标是否在区域内
    return (x_min <= x <= x_max) and (y_min <= y <= y_max)


def single_file_predict_online(save_path, area, output_dir="result_visual"):
    """单文件预测函数"""
    # 创建可视化输出目录
    os.makedirs(output_dir, exist_ok=True)
    x, y = 0, 0
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
        mask = cv2.imread(save_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError("无法读取图像文件")

        # 形态学闭运算
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 2. 实际面积计算
        actual_area = area
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
        ellipse = cv2.fitEllipseDirect(largest_contour)
        (center, (width, height), angle) = ellipse

        # 椭圆中心坐标
        x, y = center
        is_center = is_center_region(x, y)
        if is_center:
            # 动态误差补偿
            a = max(width, height) / 2.0 * 0.92
            b = min(width, height) / 2.0 * 0.92
            ellipse_area = np.pi * a * b

            # 记录预测结果
            result['predicted_area'] = ellipse_area
            result['error_percent'] = abs((ellipse_area - actual_area) / actual_area) * 100
            result['axes'] = (a * 2, b * 2)  # 返回完整轴长

            error = abs((ellipse_area - actual_area) / actual_area) * 100

            print(f"处理：实际={actual_area}, 拟合={ellipse_area:.1f}, 误差={error:.2f}%,a={a},b={b}")
            # 4. 生成可视化结果
            ellipse_adj = (center,
                           (width * 0.92, height * 0.92),  # 轴长修正
                           angle)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            FILE_DIR = os.path.join(os.path.dirname(__file__), "captured_images")
            save_path = os.path.join(FILE_DIR, filename)
            visualize_result(mask, ellipse, ellipse_adj, actual_area, save_path)

            return True, a, b, ellipse_perimeter(a, b), error, x, y

        else:
            return False, -1, -1, -1, -1, x, y

    except Exception as e:
        result['status'] = 'error'
        result['message'] = str(e)
        print(e)
        return False, -1, -1, -1, -1, x, y


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


def visualize_result(orig_mask, ellipse, ellipse_adj, actual_area, save_path):
    """可视化函数（与原函数相同）"""
    vis = cv2.cvtColor(orig_mask, cv2.COLOR_GRAY2BGR)
    # 计算参数
    (cx, cy), (w_raw, h_raw), angle = ellipse
    (_, _), (w_adj, h_adj), _ = ellipse_adj
    # 计算两种椭圆面积
    area_raw = np.pi * (w_raw / 2) * (h_raw / 2)
    area_adj = np.pi * (w_adj / 2) * (h_adj / 2)
    # cv2.drawContours(vis, [contour], -1, (0, 255, 0), 1)
    # 颜色配置（BGR格式）
    COLOR_RAW = (50, 50, 255)  # 亮红色（原始拟合）
    COLOR_ADJ = (50, 255, 50)  # 荧光绿（修正拟合）
    COLOR_ACTUAL = (255, 255, 255)  # 白色（实际值）
    COLOR_AXIS = (200, 200, 200)  # 浅灰色坐标系

    # 绘制椭圆（调整线宽和样式）
    cv2.ellipse(vis, ellipse, COLOR_RAW,
                thickness=1, lineType=cv2.LINE_AA)  # 原始拟合
    cv2.ellipse(vis, ellipse_adj, COLOR_ADJ,
                thickness=1, lineType=cv2.LINE_AA)  # 修正拟合

    # 计算误差率
    error_raw = abs((area_raw - actual_area) / actual_area) * 100
    error_adj = abs((area_adj - actual_area) / actual_area) * 100
    # text = [
    #     f"Actual: {actual}",
    #     f"Predicted: {pred:.1f}",
    #     f"Error: {error:.2f}%"
    # ]
    # for i, t in enumerate(text):
    #     y = 30 + i * 30
    #     cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # cv2.imwrite(save_path, vis)
    # 添加参数标注
    y_offset = 220
    cv2.putText(vis, f"Raw: {area_raw:.1f}px ({error_raw:.1f}%)", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
    cv2.putText(vis, f"Adj: {area_adj:.1f}px ({error_adj:.1f}%)", (10, y_offset + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
    cv2.putText(vis, f"Actual: {actual_area}px", (10, y_offset + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

    # # 添加图例
    cv2.line(vis, (int(cx) - 45, int(cy)), (int(cx) + 45, int(cy)), (200, 200, 200), 1)
    cv2.line(vis, (int(cx), int(cy) - 70), (int(cx), int(cy) + 70), (200, 200, 200), 1)

    cv2.imwrite(save_path, vis)


# 使用示例
if __name__ == "__main__":
    # # 测试单文件预测
    # test_file = "captured_images/capture_20250514_212240.jpg_debug.png"  # 修改为实际文件路径
    # result = single_file_predict(test_file)
    #
    # # 打印结果
    # if result['status'] == 'success':
    #     print("======== 预测结果 ========")
    #     print(f"实际面积: {result['actual_area']} px²")
    #     print(f"预测面积: {result['predicted_area']:.1f} px²")
    #     print(f"长轴长度: {result['axes'][0]:.1f} px")
    #     print(f"短轴长度: {result['axes'][1]:.1f} px")
    #     print(f"面积误差: {result['error_percent']:.2f}%")
    #     print(f"可视化结果: {result['vis_path']}")
    # else:
    #     print(f"处理失败: {result['message']}")
    # 测试点1：中心点 (159.5, 159.5)
    print(is_center_region(159.5, 159.5))  # 输出 True

    # 测试点2：边缘点 (47, 47)
    print(is_center_region(47, 47))  # 输出 True

    # 测试点3：角落点 (0, 0)
    print(is_center_region(0, 0))  # 输出 False
