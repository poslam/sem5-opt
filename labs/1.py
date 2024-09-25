import matplotlib.pyplot as plt
import numpy as np
from sympy import solve, symbols

l = 10**-4
ksi = 10**-5

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

A = np.dot(B.T, B)

b = np.array([1, 2, 3, 14, 5, 6]).T
x0 = np.array([1, 1, 2, 1, 1, 1]).T

f = lambda x: 0.5 * np.dot(np.dot(x.T, A), x) + np.dot(b, x)
f_d = lambda x: 0.5 * np.dot((A.T + A), x) + b

# проверка пол. определенности

eigenvalues = np.linalg.eig(A)[0]
B_det = np.linalg.det(B)

# точное

x = symbols("x0 x1 x2 x3 x4 x5")

solution = list(solve(f_d(x), x).values())

# градиент

xk = x0
xk1 = xk - l * f_d(xk)

norm = np.linalg.norm(xk1 - xk)

xk_array = [xk]
f_values = [f(xk)]

while norm >= ksi:
    xk = xk1
    xk1 = xk - l * f_d(xk)

    norm = np.linalg.norm(xk1 - xk)

    xk_array.append(xk)
    f_values.append(f(xk))

steps = len(xk_array)

# промежуточные результаты

pr = [
    (xk_array[i], f(xk_array[i]))
    for i in [
        steps // 4,
        steps // 2,
        steps // 4 * 3,
        steps - 1,
    ]
]

# x*

x_a = -np.dot(np.linalg.inv(A), b)
f_a = f(x_a)

# delta

delta_x = [abs(xk[i] - solution[i]) for i in range(len(xk))]
delta_f = abs(f(xk) - f_a)

# print

print(
    f"""
A: {A}
eigvals: {eigenvalues}
B_det: {B_det}
A_is_correct: {all([i > 0 for i in eigenvalues]) and B_det != 0}
steps: {steps}
xk: {xk}
solution: {solution}
f_a: {f_a}
pr: {pr}
delta x: {delta_x}
delta f: {delta_f}
      """
)

# img

plt.plot(range(len(xk_array)), f_values)
plt.show()
