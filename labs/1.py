from datetime import datetime

import numpy as np
from sympy import Matrix, lambdify, solve, symbols

l = 10 ** (-4)
ksi = 10 ** (-5)

B = np.array(
    [
        [9, 13, 5, 2, 5, 7],
        [1, 11, 7, 6, 1, 8],
        [3, 7, 4, 1, 3, 5],
        [6, 0, 7, 10, 6, 0],
        [1, 3, 5, 2, 11, 3],
        [5, 7, 3, 0, 5, 9],
    ]
)

A = Matrix(B.T @ B)
b = Matrix([1, 2, 3, 14, 5, 6]).T
x0 = np.array([1, 1, 2, 1, 1, 1])

x = Matrix(symbols("x:6"))

f = 0.5 * x.T * A * x + b * x

grad_f = 0.5 * (A.T + A) * x + b.T
grad_f_l = lambda x: 0.5 * (A.T + A) * x + b.T

solution = solve(grad_f, x)

# print(
#     solution,
#     "\n",
#     f.subs(solution)[0],
#     "\n",
#     lambdify(x, f)(*x0)[0][0],
# )

xk = x0
xk1 = xk - l * grad_f_l(xk)

print(xk, xk1)
# norm = np.linalg.norm(xk1 - xk)

xk_array = [xk]

# while norm >= ksi:
#     xk = xk1
#     xk1 = xk - l * grad_f_l(xk)

#     norm = np.linalg.norm(xk1 - xk)

#     xk_array.append(xk)

#     print(len(xk_array), norm)

#     if len(xk_array) > 1000:
#         break
