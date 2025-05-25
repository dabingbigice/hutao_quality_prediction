import math

import numpy as np
from scipy.optimize import fsolve
from scipy.special import ellipe
def calculate_aspect_ratio(perimeter, area):
    # 计算长轴 L 和短轴 W
    discriminant = perimeter ** 2 - 16 * area
    if discriminant < 0:
        return None  # 无实数解（可能不是矩形）

    sqrt_discriminant = math.sqrt(discriminant)
    L1 = (perimeter + sqrt_discriminant) / 4
    L2 = (perimeter - sqrt_discriminant) / 4

    # 取较大的值作为 L（长轴）
    L = max(L1, L2)
    print(f'长:{L}')
    W = area / L  # 因为 A = L × W
    print(f'宽:{W}')

    aspect_ratio = L / W
    return L, W, aspect_ratio

def ellipse_perimeter(a, b):
    if a < b:
        a, b = b, a  # 确保 a >= b
    e_sq = 1 - (b ** 2 / a ** 2)
    return 4 * a * ellipe(e_sq)
def ellipse_axes_from_area_perimeter(A, P):
    # 检查最小周长
    min_P = 2 * np.pi * np.sqrt(A / np.pi)
    if P < min_P:
        raise ValueError(f"周长 P = {P} 太小，至少需要 {min_P:.2f}")

    def equations(vars):
        a, b = vars
        eq1 = np.pi * a * b - A
        eq2 = ellipse_perimeter(a, b) - P
        return [eq1, eq2]

    # 初始猜测：假设 a >= b
    initial_a = np.sqrt(A)
    initial_b = A / (np.pi * initial_a)
    a, b = fsolve(equations, (initial_a, initial_b))

    return a, b, a / b
