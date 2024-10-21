import numpy as np

eps = 1e-6


def f(A, x, b):
    return 1 / 2 * x.T @ A @ x + b @ x


def check_matrix(A):
    def check_symmetric(a, tol=1e-8):
        return np.allclose(a, a.T, atol=tol)

    def check_non_singular(a):
        return np.linalg.det(a) != 0

    return check_symmetric(A) and check_non_singular(A)


def jacobian(A: np.ndarray, x0: np.ndarray, x: np.ndarray):
    n = len(x0)
    J = np.zeros((n + 1, n + 1))
    J[:n, :n] = A + 2 * x[-1] * np.eye(n)
    J[:n, n] = 2 * (x[:n] - x0)
    J[n, :n] = 2 * (x[:n] - x0)
    return J


def F(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    x: np.ndarray,
    r: int,
) -> np.ndarray:
    result = A @ x[: len(x0)] + 2 * x[-1] * (x[: len(x0)] - x0) + b
    result = np.append(result, np.linalg.norm(x[: len(x0)] - x0) ** 2 - r**2)
    return result


def newton(A, x0, r, b, start, eps=eps):
    xk = [start]

    f_val = F(A, b, x0, start, r)
    j = jacobian(A, x0, start)
    rev_j = np.linalg.inv(j)
    xk.append(xk[0] - rev_j @ f_val)

    while np.linalg.norm(xk[-1] - xk[-2]) > eps:
        f1 = F(A, b, x0, xk[-1], r)
        j1 = jacobian(A, x0, xk[-1])
        rev_j1 = np.linalg.inv(j1)
        xk.append(xk[-1] - rev_j1 @ f1)

    return xk[-1]


A = np.array(
    [
        [26.20056955320149, -58.76847438658791, 414.685599118923, 254.48571079046798],
        [-58.76847438658791, 945.449926413821, 355.0397007546424, 487.73320892614504],
        [414.685599118923, 355.0397007546424, -604.0485273715911, -423.5778583766898],
        [254.48571079046798, 487.73320892614504, -423.5778583766898, 257.4102340464],
    ]
)
b = np.array([1, 2, 3, 4])
x0 = np.array([1, 1, 1, 1])
r = 5

# if y = 0

x_min = -np.linalg.inv(A) @ b

# if y > 0

starts = [
    np.array([0.1, 0.2, 0.3, 0.4, r]),
    np.array([0.6, 0.7, 0.8, 0.9, r]),
    np.array([1.1, 1.2, 1.3, 1.4, r]),
    np.array([1.6, 1.7, 1.8, 1.9, r]),
    np.array([2.1, 2.2, 2.3, 2.4, r]),
    np.array([2.6, 2.7, 2.8, 2.9, r]),
    np.array([3.1, 3.2, 3.3, 3.4, r]),
    np.array([3.6, 3.7, 3.8, 3.9, r]),
]

res = [newton(A=A, x0=x0, r=r, b=b, start=start, eps=eps) for start in starts]

# print

print(
    f"""
check matrix A if it's symmetric and not singular: {check_matrix(A)}

solution (if y = 0): 

x = {x_min}
f = {f(A, x_min, b)}
norm = {np.linalg.norm(x_min - x0)}
r = {r}
norm <= r? {np.linalg.norm(x_min) <= r}

solution (if y > 0):
"""
)

for i in range(len(res)):
    print(f"{i+1} \t f: {f(A, res[i][:4], b)} \t x: {res[i]}")