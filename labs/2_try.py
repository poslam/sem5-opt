import numpy as np

# Constants
A = np.array([
    [26.20056955320149, -58.76847438658791, 414.685599118923, 254.48571079046798],
    [-58.76847438658791, 945.449926413821, 355.0397007546424, 487.73320892614504],
    [414.685599118923, 355.0397007546424, -604.0485273715911, -423.5778583766898],
    [254.48571079046798, 487.73320892614504, -423.5778583766898, 257.4102340464]
])
b = np.array([1, 2, 3, 4])
x0 = np.array([1, 1, 1, 1])
r = 4
eps = 1e-6

# Functions
def f_1(x, y):
    return np.concatenate([
        (A + 2 * np.identity(4) * y) @ x[:4] + (b + 2 * y * x0),
        np.array([np.linalg.norm(x[:4] - x0) ** 2 - r**2])
    ])

def f_1_diff(x, y):
    return np.block([
        [A + 2 * np.identity(4) * y, 2 * (x[:4] - x0).reshape(-1, 1)],
        [2 * (x[:4] - x0).reshape(1, -1), np.array([[0]])]
    ])

# Initial guess
x_init = np.array([0.1, 0.2, 0.3, 0.4, 1.0])

# Newton's method
def newton_method(f, f_diff, x_init, y_init, eps):
    xk = x_init
    yk = y_init
    xk1 = xk - np.linalg.inv(f_diff(xk, yk)) @ f(xk, yk)
    norm = np.linalg.norm(xk1 - xk)

    while norm > eps:
        xk = xk1
        xk1 = xk - np.linalg.inv(f_diff(xk, yk)) @ f(xk, yk)
        norm = np.linalg.norm(xk1 - xk)

    return xk1

# Find solution
solution = newton_method(f_1, f_1_diff, x_init, 1, eps)
print("Solution:", solution)