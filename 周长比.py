import numpy as np
from scipy.optimize import fsolve
from scipy.special import ellipe


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

    return a, b


# 示例
A = 2815
P = 25  # 必须 >= 2π√(A/π) ≈ 19.42
a, b = ellipse_axes_from_area_perimeter(A, P)
print(f"长半轴 a = {a:.2f}, 短半轴 b = {b:.2f}")