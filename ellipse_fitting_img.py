import cv2
import numpy as np
import os
from datetime import datetime
import math


def ellipse_perimeter(a, b):
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


def single_file_predict(image_path, area, output_dir="result_visual"):
    # """整合高精度轮廓检测的改进版函数"""
    # # 初始化结果字典
    # result = {
    #     'status': 'success',
    #     'message': '',
    #     'actual_area': 0,
    #     'predicted_area': 0,
    #     'error_percent': 0.0,
    #     'axes': (0.0, 0.0),
    #     'vis_path': ''
    # }
    # a, b = 0.0, 0.0  # 显式初始化
    #
    # try:
    #     # 1. 多级预处理
    #     orig_mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #     if orig_mask is None:
    #         raise ValueError("无法读取图像文件")
    #
    #     # 增强对比度 (CLAHE)
    #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #     enhanced = clahe.apply(orig_mask)
    #
    #     # 复合形态学处理
    #     kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #     opened = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel_open, iterations=1)
    #
    #     kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    #     closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=3)
    #
    #     # 2. 高精度边缘检测
    #     smoothed = cv2.bilateralFilter(closed, d=9, sigmaColor=75, sigmaSpace=75)
    #     edges = cv2.Canny(smoothed, 30, 100)  # 调整Canny阈值
    #
    #     # 3. 优化轮廓检测
    #     contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
    #     if not contours:
    #         raise ValueError("未检测到有效轮廓")
    #
    #     # 亚像素级轮廓优化
    #     max_contour = max(contours, key=cv2.contourArea)
    #     contour_subpix = cv2.cornerSubPix(
    #         smoothed,
    #         np.float32(max_contour.reshape(-1, 2)),
    #         winSize=(5, 5),  # 增大窗口尺寸
    #         zeroZone=(-1, -1),
    #         criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
    #     )
    #
    #     # 4. 椭圆拟合与验证
    #     ellipse = cv2.fitEllipse(contour_subpix)
    #     (_, (width, height), _) = ellipse
    #
    #     # 动态补偿系数
    #     a = max(width, height) / 2.0 * 0.982
    #     b = min(width, height) / 2.0 * 0.986
    #
    #     # 计算实际面积
    #     actual_area = cv2.contourArea(contour_subpix)
    #     ellipse_area = np.pi * a * b
    #
    #     # 记录结果
    #     result.update({
    #         'actual_area': actual_area,
    #         'predicted_area': ellipse_area,
    #         'error_percent': abs((ellipse_area - actual_area) / actual_area * 100),
    #         'axes': (a * 2, b * 2)
    #     })
    #
    #     # 5. 可视化增强
    #     debug_img = cv2.cvtColor(orig_mask, cv2.COLOR_GRAY2BGR)
    #     cv2.drawContours(debug_img, [contour_subpix.astype(int)], -1, (0, 255, 0), 2)
    #     cv2.ellipse(debug_img, ellipse, (0, 0, 255), 2)
    #
    #     # 保存结果
    #     os.makedirs(output_dir, exist_ok=True)
    #     vis_path = os.path.join(output_dir, f"enhanced_{os.path.basename(image_path)}")
    #     cv2.imwrite(vis_path, debug_img)
    #     result['vis_path'] = vis_path
    #
    #     return result,a,b
    #
    # except Exception as e:
    #     result.update({
    #         'status': 'error',
    #         'message': str(e)
    #     })
    #     return result,a,b
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
        visualize_result(mask, ellipse,ellipse_adj,actual_area, vis_path)
        result['vis_path'] = vis_path
        ellipse_perimeter(a, b)
        return result, a, b, ellipse_perimeter(a, b)

    except Exception as e:
        result['status'] = 'error'
        result['message'] = str(e)
        print(e)
        return result, a, b, ellipse_perimeter(a, b)


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


def visualize_result(orig_mask, ellipse,ellipse_adj,actual_area, save_path):
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
    error_raw = abs((area_raw - actual_area)/actual_area)*100
    error_adj = abs((area_adj - actual_area)/actual_area)*100
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
    # 测试单文件预测
    test_file = "captured_images/capture_20250514_212240.jpg_debug.png"  # 修改为实际文件路径
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
