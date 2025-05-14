import cv2
import numpy as np
from skimage.measure import EllipseModel, ransac


def ransac_fit_ellipse(contour, max_trials=100, residual_threshold=3.0):
    """
    使用 RANSAC 拟合椭圆（正确版本）
    :param contour: 轮廓点集，形状为 (N, 1, 2)
    :param max_trials: 最大迭代次数
    :param residual_threshold: 内点误差阈值（单位：像素）
    :return: 椭圆参数 (中心点, 长宽, 旋转角)
    """
    # 转换输入格式
    points = np.squeeze(contour).astype(np.float32)

    # 使用 skimage 的 ransac 函数
    model, inliers = ransac(
        points,
        EllipseModel,
        min_samples=6,
        residual_threshold=residual_threshold,
        max_trials=max_trials,
        random_state=42
    )

    # 提取椭圆参数
    center = (model.params[0], model.params[1])
    width = model.params[2] * 2
    height = model.params[3] * 2
    angle = np.degrees(model.params[4]) % 180

    return (center, (width, height), angle)


# 示例使用
mask = cv2.imread("hutao.png", cv2.IMREAD_GRAYSCALE)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
max_contour = max(contours, key=cv2.contourArea)

# 对比两种方法
dls_ellipse = cv2.fitEllipse(max_contour)  # OpenCV 默认方法
ransac_ellipse = ransac_fit_ellipse(max_contour)  # RANSAC 方法

# 可视化结果
vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
cv2.ellipse(vis, dls_ellipse, (0, 255, 0), 2)  # 绿色：默认方法
cv2.ellipse(vis, ransac_ellipse, (0, 0, 255), 2)  # 红色：RANSAC

cv2.imwrite("compare.png", vis)