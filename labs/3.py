import sys

import numpy as np
import scipy.optimize as opt

from labs.funcs import *

sys.stdout = open("./labs/output.txt", "w")


def simplex_method(tableau: np.ndarray) -> tuple[float, np.ndarray]:
    n, m = tableau.shape
    step = 1

    while np.any(tableau[0, :-1] < 0):
        pivot_col = -1
        min_value = 0
        for j in range(m - 1):
            if tableau[0, j] < min_value:
                min_value = tableau[0, j]
                pivot_col = j

        pivot_row = -1
        min_ratio = np.float64("inf")
        for i in range(1, n):
            if tableau[i, pivot_col] > 0:
                ratio = tableau[i, -1] / tableau[i, pivot_col]
                if ratio < min_ratio and ratio >= 0:
                    min_ratio = ratio
                    pivot_row = i

        if pivot_row == -1:
            raise ValueError("no solution")

        pivot_value = tableau[pivot_row, pivot_col]

        tableau[pivot_row] /= pivot_value

        for i in range(n):
            if i != pivot_row:
                tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]

        print(f"step:\t{step}\tpivot_value:\t{pivot_value}")
        print_matrix(tableau)

        step += 1

    solution = np.zeros(m - 1)
    for i in range(1, n):
        pivot_col = np.where(tableau[i, :-1] == 1)[0]
        if pivot_col.size > 0:
            solution[pivot_col[0]] = tableau[i, -1]

    return tableau[0, -1], solution


def first_way(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> tuple[float, np.ndarray]:
    print("Прямая задача:")

    n, m = A.shape

    tableau = np.zeros((n + 1, m + n + 1))

    tableau[1:, :m] = A
    tableau[1:, m : m + n] = np.eye(n)
    tableau[1:, -1:] = b
    tableau[0, :m] = -c

    print_matrix(tableau, header="Исходная таблица")

    x, result = simplex_method(tableau)
    result = result[: A.shape[1]]

    print(f"\nЦелевая функция: {x}\n")
    print_matrix(np.array([result]), header="Начальное угловое решение")

    return x, result


def additional_task(A_T, b_T, c_T):
    print("Вспомогательная задача")
    n, m = A_T.shape

    tableau = np.zeros((n + 1, m + n + n + 1))
    tableau[1:, :m] = A_T
    tableau[1:, m : m + n] = -1 * np.eye(n)
    tableau[1:, m + n : m + n + n] = np.eye(n)
    tableau[1:, -1:] = c_T
    tableau[0, m * 2 - 2 : -1] = np.array([1 for i in range(n)])

    for i in range(1, n + 1):
        tableau[0] += tableau[i] * -1

    print_matrix(tableau, header="Исходная таблица")
    x, result = simplex_method(tableau)

    print_matrix(np.array([result]), header="Начальное угловое решение")
    print(f"Целевая функция: {x}")

    return tableau, b_T


def second_way(A, b, c):
    A_T = A.T
    b_T = b.T
    c_T = c.T

    n, m = A_T.shape

    tableau, b_T = additional_task(A_T, b_T, c_T)

    print("\n\n", "═" * 500, "\n", "═" * 500, "\n", "═" * 500, "\n\n")

    print("Двойственная задача")

    tableau = np.hstack((tableau[:, : m + n], tableau[:, -1:]))
    filler_b = np.array([[0 for _ in range(tableau.shape[1] - b_T.shape[1])]])
    filler_b = np.hstack((b_T, filler_b))

    tableau[0] = filler_b

    print_matrix(tableau, header="Исходная таблица")

    for i in range(1, n + 1):
        tableau[0] += -tableau[i]

    print_matrix(tableau, header="next")

    x, result = simplex_method(tableau)

    print_matrix(np.array([result]), header="Решение")
    print(f"Целевая функция: {x}")


# def second_way(A, b, c):
#     # Для задачи максимизации:
#     # Прямая:    max c^T x
#     #           s.t. Ax <= b

#     # Двойственная должна быть:
#     # min b^T y
#     # s.t. A^T y >= c
#     # y >= 0

#     A_T = A.T
#     b_T = b.T
#     c_T = c.T

#     n, m = A_T.shape

#     # Нужно изменить знаки неравенств и целевую функцию:
#     tableau = np.zeros((n + 1, m + n + n + 1))
#     tableau[1:, :m] = -A_T  # Меняем знак матрицы
#     tableau[0, :m] = b_T  # Коэффициенты новой целевой функции
#     tableau[1:, m : m + n] = np.eye(n)
#     # tableau[1:, -1] = -c_T  # Меняем знак правых частей
#     tableau[1:, -1:] = -c_T

#     print_matrix(tableau, header="Исходная таблица двойственной задачи")

#     x, result = simplex_method(tableau)

#     print(x)

#     # Результат нужно взять с обратным знаком
#     return -x, -result


### data --------------------------------------------------------------------------

A = np.array(
    [
        [15, 115, 106, 290, 232, 167],
        [79, 247, 7, 286, 65, 276],
        [219, 125, 174, 42, 114, 202],
        [287, 213, 225, 274, 169, 260],
        [202, 124, 211, 200, 174, 183],
        [158, 265, 1, 39, 113, 290],
        [175, 196, 170, 270, 187, 178],
        [245, 100, 226, 63, 245, 259],
    ]
)

b = np.array(
    [
        [296],
        [85],
        [22],
        [47],
        [247],
        [28],
        [125],
        [218],
    ]
)

c = np.array([[173, 299, 240, 120, 249, 86]])

# low, high = 0, 100

# A = np.random.uniform(low, high, (8, 6))

# b = np.random.uniform(low, high, (8, 1))

# c = np.random.uniform(low, high, (1, 6))


### прямая задача ------------------------------------------------------------------

first_way(A, b, c)

print("\n", "═" * 500, "\n", "═" * 500, "\n", "═" * 500, "\n\n")

### двойственная задача ------------------------------------------------------------

second_way(A, b, c)
