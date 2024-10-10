import numpy as np


def check_matrix(A):
    def check_symmetric(a, tol=1e-8):
        return np.allclose(a, a.T, atol=tol)

    def check_non_degenerate(a):
        return np.linalg.det(a) != 0

    return check_symmetric(A) and check_non_degenerate(A)


A = np.array(
    [
        [26.20056955320149, -58.76847438658791, 414.685599118923, 254.48571079046798],
        [-58.76847438658791, 945.449926413821, 355.0397007546424, 487.73320892614504],
        [414.685599118923, 355.0397007546424, -604.0485273715911, -423.5778583766898],
        [254.48571079046798, 487.73320892614504, -423.5778583766898, 257.4102340464],
    ]
)
b = np.array([1, 2, 3, 4]).T
x0 = np.array([1, 1, 1, 1]).T
r = 4


print(A, b, x0, r)
