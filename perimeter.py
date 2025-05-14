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
    mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (320, 320), interpolation=cv2.INTER_AREA)

    max_contour, perimeter = calculate_perimeter(mask)
    if max_contour is not None:
        print(f"Perimeter: {perimeter} pixels")

        # 创建彩色画布
        img_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_color, [max_contour], -1, (0, 255, 0), 2)

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