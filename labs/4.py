"""
for max:
index_of_element = argmax()
... <= 0: break
return -
c
"""

import sys

import numpy as np
from funcs import print_matrix, print_matrix_latex

sys.stdout = open("./labs/output.txt", "w", encoding="utf-8")


def make_matrix(A: np.ndarray, b: np.ndarray, c: np.ndarray):
    return np.vstack(
        (
            np.hstack((np.reshape(b, (A.shape[0], 1)), A, np.eye(A.shape[0]))),
            np.hstack(((np.array([0])), c, np.zeros((A.shape[0])))),
        )
    )


def make_dual_matrix(A: np.ndarray, b: np.ndarray, c: np.ndarray):
    return np.vstack(
        (
            np.hstack((np.reshape(c, (A.T.shape[0], 1)), -A.T, np.eye(A.T.shape[0]))),
            np.hstack(((np.array([0])), -b, np.zeros((A.T.shape[0])))),
        )
    )


def simplex(simplex_matrix: np.ndarray, n: int, m: int):
    while True:
        index_of_element = simplex_matrix[-1, 1:].argmin()

        if simplex_matrix[-1, 1:][index_of_element] >= 0:
            break

        else:
            min_element = np.inf
            min_line = 0
            index_of_element += 1

            for line in range(simplex_matrix.shape[0] - 1):
                if (
                    simplex_matrix[line, index_of_element] > 0
                    and simplex_matrix[line, 0]
                    / simplex_matrix[
                        line,
                        index_of_element,
                    ]
                    < min_element
                ):
                    min_line = line
                    min_element = (
                        simplex_matrix[line, 0]
                        / simplex_matrix[
                            line,
                            index_of_element,
                        ]
                    )

            print(
                f"Индекс: {(min_line, int(index_of_element))}\n"
                # + f"focus func val: {simplex_matrix[-1, int(index_of_element)]:.3f}\n"
                + f"Разрешающий элемент: {simplex_matrix[min_line, int(index_of_element)]:.3f}",
            )
            print_matrix(simplex_matrix)

            simplex_matrix[min_line, :] = (
                simplex_matrix[min_line, :]
                / simplex_matrix[
                    min_line,
                    index_of_element,
                ]
            )

            for line in range(simplex_matrix.shape[0]):
                if line == min_line:
                    continue

                simplex_matrix[line, :] = (
                    simplex_matrix[line, :]
                    - simplex_matrix[min_line, :]
                    * simplex_matrix[line, index_of_element]
                )

    ans = np.zeros(m)

    for i in range(n - 1):
        for j in range(1, m + 1):
            if simplex_matrix[i, j] == 1:
                ans[j - 1] = simplex_matrix[i, 0]
                break

    print("result: ")
    print_matrix(simplex_matrix)

    return simplex_matrix[-1, 0], simplex_matrix, ans


def dual_simplex(simplex_matrix: np.ndarray, n: int, m: int):
    while True:
        index_of_element = simplex_matrix[:-1, 0].argmin()

        if simplex_matrix[:-1, 0][index_of_element] >= 0:
            break

        else:
            min_element = np.inf
            min_column = 0

            for column in range(1, simplex_matrix.shape[1]):
                if simplex_matrix[-1, column] == 0:
                    continue

                if (
                    simplex_matrix[index_of_element, column] < 0
                    and abs(
                        simplex_matrix[-1, column]
                        / simplex_matrix[index_of_element, column]
                    )
                    < min_element
                ):
                    min_column = column
                    min_element = abs(
                        simplex_matrix[-1, column]
                        / simplex_matrix[index_of_element, column]
                    )

            print(
                f"Индекс: {(int(index_of_element), min_column)}\n"
                # + f"focus func val: {simplex_matrix[:-1, 0][index_of_element]:.3f}\n"
                + f"Разрешающий элемент: {simplex_matrix[int(index_of_element), min_column]:.3f}",
            )
            print_matrix(simplex_matrix)

            simplex_matrix[index_of_element, :] /= simplex_matrix[
                index_of_element, min_column
            ]

            for line in range(simplex_matrix.shape[0]):
                if line == index_of_element:
                    continue

                simplex_matrix[line, :] -= (
                    simplex_matrix[index_of_element, :]
                    * simplex_matrix[line, min_column]
                )

    ans = np.zeros(m)

    for i in range(n - 1):
        for j in range(1, m + 1):
            if simplex_matrix[i, j] == 1:
                ans[j - 1] = simplex_matrix[i, 0]
                break

    print("result: ")
    print_matrix(simplex_matrix)

    return simplex_matrix[-1, 0], simplex_matrix, ans


A = np.array(
    [
        [0, -16, -41, 48, 19, 84, 69, 33],
        [82, 98, -50, 84, -52, -47, -95, -20],
        [65, 12, 61, -88, -18, -85, 34, -10],
        [72, 37, 9, 28, 33, -31, 85, 18],
        [32, -24, -70, -70, 53, 60, 22, 60],
        [12, -37, 53, 81, -34, 21, -29, -67],
    ]
)

print_matrix(A)

tmp = []

for i in range(A.shape[0]):
    tmp.append(min(A[i, :]))

print("Нижняя цена игры:", max(tmp))

tmp.clear()

for i in range(A.shape[1]):
    tmp.append(max(A[:, i]))

print("Верхняя цена игры:", min(tmp), "\n")

beta = A.min()

A_cap: np.ndarray = A + np.abs(beta)

print_matrix(A_cap, header="A_cap")

print("beta: ", beta, "\n")

b = np.ones(A_cap.shape[0])
c = np.ones(A_cap.shape[1])

print("simplex", end="\n\n")

x1 = simplex(
    make_matrix(A_cap, b, -c),
    n=A_cap.shape[0],
    m=A_cap.shape[1],
)

print("dual simplex", end="\n\n")

x2 = dual_simplex(
    make_dual_matrix(A_cap, b, -c),
    n=A_cap.shape[1],
    m=A_cap.shape[0],
)

print_matrix(x1[2] / np.linalg.norm(x1[2]), "Оптимальная стратегия первого игрока")
print_matrix(x2[2] / np.linalg.norm(x2[2]), "Оптимальная стратегия второго игрока")

if np.abs(x1[0] - x2[0]) > 1e-15:
    raise ValueError("straight != dual")

print(f"alpha: {x1[0]:.5f}")
print(f"Цена игры: {1 / x1[0] - np.abs(beta):.5f}")
