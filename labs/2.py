import numpy as np
import sympy as sp

from other.script import matrices_to_word

eps = 1e-6


def check_matrix(A):
    def check_symmetric(a, tol=1e-8):
        return np.allclose(a, a.T, atol=tol)

    def check_non_degenerate(a):
        return np.linalg.det(a) != 0

    return check_symmetric(A) and check_non_degenerate(A)


def solve_y_eq_0(A, b, x0, r, f):
    x_min = -np.linalg.inv(A) @ b

    norm_to_check = np.linalg.norm(x_min - x0)

    if norm_to_check > r:
        raise Exception("norm(x-x0) > r")

    return {"x": x_min, "f": f(x_min), "norm": norm_to_check}


A = np.array(
    [
        [26.20056955320149, -58.76847438658791, 414.685599118923, 254.48571079046798],
        [-58.76847438658791, 945.449926413821, 355.0397007546424, 487.73320892614504],
        [414.685599118923, 355.0397007546424, -604.0485273715911, -423.5778583766898],
        [254.48571079046798, 487.73320892614504, -423.5778583766898, 257.4102340464],
    ]
)
b = np.array([1, 2, 3, 4]).reshape(-1, 1)
x0 = np.array([1, 1, 1, 1]).reshape(-1, 1)
r = 4

f = lambda x: 0.5 * x.T @ A @ x + b.T @ x

L = lambda x, y: 0.5 * x.T @ A @ x + b @ x + y * (np.linalg.norm(x - x0) ** 2 - r**2)
L_diff_x = lambda x, y: A @ x + b + 2 * y @ (x - x0)

# y = 0

x_min, f_res, norm = solve_y_eq_0(A, b, x0, r, f).values()

# y > 0

# f_1 = lambda x, y: np.array(
#     [
#         [(A + 2 * np.identity(4) @ y) @ x + (b + 2 * y @ x0), 0],
#         [np.linalg.norm(x - x0) ** 2 - r**2, 0],
#     ]
# )


f_1 = lambda x, y: np.concatenate(
    [
        (A + 2 * np.identity(4) * y) @ x[:4] + (b + 2 * y * x0),
        np.array([np.linalg.norm(x[:4] - x0) ** 2 - r**2]),
    ]
)

# f_1_diff = lambda x, y: np.array(
#     [
#         [A + 2 * np.identity(4) * y, 2 * (x - x0)],
#         [2 * (x - x0).T, 0],
#     ]
# )

f_1_diff = lambda x, y: np.block(
    [
        [A + 2 * np.identity(4) * y, 2 * (x - x0).reshape(-1, 1)],
        [2 * (x - x0).reshape(1, -1), 0],
    ]
)

vectors = [
    np.array([0.1, 0.2, 0.3, 0.4]),
    np.array([0.6, 0.7, 0.8, 0.9]),
    np.array([1.1, 1.2, 1.3, 1.4]),
    np.array([1.6, 1.7, 1.8, 1.9]),
    np.array([2.1, 2.2, 2.3, 2.4]),
    np.array([2.6, 2.7, 2.8, 2.9]),
    np.array([3.1, 3.2, 3.3, 3.4]),
    np.array([3.6, 3.7, 3.8, 3.9]),
]


for x0 in vectors:
    xk = x0
    xk1 = xk - np.linalg.inv(f_1_diff(xk, r)) @ f_1(xk, r)

    norm = np.linalg.norm(xk1 - xk)

    while norm > eps:
        xk = xk1
        xk1 = xk - np.linalg.inv(f_1_diff(xk)) @ f_1(xk)

        norm = np.linalg.norm(xk1 - xk)

    print(xk1)
