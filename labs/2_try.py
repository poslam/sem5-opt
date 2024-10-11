import numpy as np
import sympy as sp

A = sp.Matrix(
    [
        [26.20056955320149, -58.76847438658791, 414.685599118923, 254.48571079046798],
        [-58.76847438658791, 945.449926413821, 355.0397007546424, 487.73320892614504],
        [414.685599118923, 355.0397007546424, -604.0485273715911, -423.5778583766898],
        [254.48571079046798, 487.73320892614504, -423.5778583766898, 257.4102340464],
    ]
)
b = sp.Matrix([1, 2, 3, 4]).T
x0 = sp.Matrix([1, 1, 1, 1]).T
r = 4

# f_1_diff = lambda x, y: np.array(
#     [
#         [A + 2 * np.identity(4) * y, 2 * (x - x0)],
#         [2 * (x - x0).T, 0],
#     ]
# )

# f_1 = lambda x, y: np.array(
#     [
#         [(A + 2 * np.identity(4) * y) @ x + (b + 2 * y * x0), 0],
#         [np.linalg.norm(x - x0) ** 2 - r**2, 0],
#     ]
# )

# print(f_1(x0, r))


def norm(x):
    return sp.sqrt(x.dot(x))


x = sp.symbols("x0 x1 x2 x3 x4")
y = sp.Symbol("y")

sp.Matrix(
    [
        [(A + 2 * sp.Identity(4) * y) @ x + (b + 2 * y * x0), 0],
        [norm(x - x0) ** 2 - r**2, 0],
    ]
)
