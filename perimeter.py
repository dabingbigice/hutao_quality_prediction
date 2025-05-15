import cv2
import numpy as np


def calculate_perimeter(mask):
    """
    计算二值掩膜的最大轮廓及其周长
    :param mask: 输入的二值化掩膜（0为背景，255为目标）
    :return: (max_contour, perimeter) 最大轮廓对象和周长（像素单位）
    """
    # 查找所有外轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return None, 0.0  # 无轮廓时返回空

    # 按面积排序获取最大轮廓
    max_contour = max(contours, key=cv2.contourArea)

    # 计算周长
    perimeter = cv2.arcLength(max_contour, closed=True)
    return max_contour, perimeter


def hutao_perimeter(filepath):
    # upscale_factor=2
    # """高精度周长预测函数"""
    # # 1. 多尺度输入
    # orig_mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    #
    # # 2. 亚像素级缩放（保留高频信息）
    # resized = cv2.resize(orig_mask,
    #                      (orig_mask.shape[1] * upscale_factor, orig_mask.shape[0] * upscale_factor),
    #                      interpolation=cv2.INTER_CUBIC)
    #
    # # 3. 边缘保留滤波
    # smoothed = cv2.bilateralFilter(resized, d=9, sigmaColor=75, sigmaSpace=75)
    #
    # # 4. 亚像素边缘检测
    # edges = cv2.Canny(smoothed, 50, 150)
    # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # if not contours:
    #     return 0.0
    #
    # # 5. 亚像素轮廓优化
    # max_contour = max(contours, key=cv2.contourArea)
    # contour_subpix = cv2.cornerSubPix(
    #     smoothed,
    #     np.float32(max_contour.reshape(-1, 2)),
    #     winSize=(3, 3),
    #     zeroZone=(-1, -1),
    #     criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    # )
    #
    # # 6. 多尺度周长融合
    # perimeter_highres = cv2.arcLength(contour_subpix, closed=True)
    # perimeter_lowres = cv2.arcLength(max_contour, closed=True)
    #
    # # 7. 动态比例补偿
    # scale_ratio = 1.0 / upscale_factor
    # calibrated_perimeter = (perimeter_highres * 0.7 + perimeter_lowres * 0.3) * scale_ratio
    #
    # # 可视化保存
    # debug_img = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(debug_img, [contour_subpix.astype(int)], -1, (0, 255, 0), 2)
    # cv2.imwrite(f"{filepath}_debug.png", debug_img)
    #
    # return calibrated_perimeter,contour_subpix


    mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (320, 320), interpolation=cv2.INTER_AREA)

    max_contour, perimeter = calculate_perimeter(mask)
    if max_contour is not None:
        print(f"Perimeter: {perimeter} pixels")

        # 创建彩色画布
        img_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_color, [max_contour], -1, (0, 255, 0), 1)

        # 新增保存功能[1,4,8](@ref)
        output_path = f"{filepath}_perimeter_{perimeter:.2f}.png"  # 自定义保存路径
        cv2.imwrite(output_path, img_color)
        print(f"可视化结果已保存至: {output_path}")

    return perimeter

    # if max_contour is not None:
    #     print(f"Perimeter: {perimeter} pixels")
    #
    #     # 创建彩色画布
    #     img_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    #     cv2.drawContours(img_color, [max_contour], -1, (0, 255, 0), 2)
    #
    #     # 新增保存功能[1,4,8](@ref)
    #     output_path = f"contour_result_{perimeter:.2f}.png"  # 自定义保存路径
    #     cv2.imwrite(output_path, img_color)
    #     print(f"可视化结果已保存至: {output_path}")

    # 读取时保留原始尺寸

    # mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    # # 保持原图尺寸逻辑
    # original_size = mask.shape[:2]
    # if original_size[0] > 320:
    #     mask = cv2.resize(mask, (320, 320), interpolation=cv2.INTER_LINEAR_EXACT)
    #
    # # 高斯模糊
    # blurred = cv2.GaussianBlur(mask, (5, 5), sigmaX=1.5, sigmaY=1.5)
    #
    # # 二值化
    # _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #
    # # 提取轮廓
    # contours, _ = cv2.findContours(
    #     thresh,
    #     cv2.RETR_EXTERNAL,
    #     cv2.CHAIN_APPROX_TC89_KCOS
    # )
    #
    # if not contours:
    #     return 0
    #
    # max_contour = max(contours, key=cv2.contourArea)
    #
    # # 轮廓平滑
    # epsilon = 0.001 * cv2.arcLength(max_contour, True)
    # smoothed_contour = cv2.approxPolyDP(max_contour, epsilon, True)
    #
    # # 修正部分：使用单通道图像
    # term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    #
    # # 需要将轮廓转换为二维点坐标（N, 1, 2）-> (N, 2)
    # contour_points = smoothed_contour.reshape(-1, 2).astype(np.float32)
    #
    # # 正确调用方式（使用单通道图像）
    # cv2.cornerSubPix(
    #     thresh,  # 直接使用二值化图像（单通道）
    #     contour_points,  # 输入点需要是浮点型
    #     (5, 5),
    #     (-1, -1),
    #     term_crit
    # )
    #
    # # 将优化后的点转换回轮廓格式
    # optimized_contour = contour_points.reshape(-1, 1, 2).astype(np.int32)
    #
    # # 计算最终周长
    # perimeter = cv2.arcLength(optimized_contour, True)
    #
    # # 可视化（这里需要彩色图像）
    # img_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(img_color, [optimized_contour], -1, (0, 255, 0), 2)
    #
    # output_path = f"{filepath}_smooth_perimeter_{perimeter:.2f}.png"
    # cv2.imwrite(output_path, img_color)
    #
    # return perimeter

# 示例用法
if __name__ == "__main__":
    mask = cv2.imread("captured_images/capture_20250513_224842.jpg", cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (320, 320), interpolation=cv2.INTER_AREA)
    max_contour, perimeter = calculate_perimeter(mask)

    if max_contour is not None:
        print(f"Perimeter: {perimeter} pixels")

        # 创建彩色画布
        img_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_color, [max_contour], -1, (0, 255, 0), 2)

        # 新增保存功能[1,4,8](@ref)
        output_path = f"contour_result_{perimeter:.2f}.png"  # 自定义保存路径
        cv2.imwrite(output_path, img_color)
        print(f"可视化结果已保存至: {output_path}")

        # 显示窗口
        cv2.imshow("Contour", img_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()