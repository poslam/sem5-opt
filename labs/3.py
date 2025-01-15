"""
for max:
index_of_element = argmax()
... <= 0: break
return -
c
"""

import sys

import numpy as np

from labs.funcs import print_matrix, print_matrix_latex

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


def simplex(simplex_matrix: np.ndarray):
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
            print_matrix_latex(simplex_matrix)

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

    print("result: ")
    print_matrix_latex(simplex_matrix)

    return simplex_matrix[-1, 0], simplex_matrix


def dual_simplex(simplex_matrix: np.ndarray):
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
            print_matrix_latex(simplex_matrix)

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

    print("result: ")
    print_matrix_latex(simplex_matrix)

    return simplex_matrix[-1, 0], simplex_matrix


A = np.array(
    [
        [0, 115, 16, 29, 23, 197],
        [98, 247, 7, 28, 65, 27],
        [9, 125, 174, 421, 14, 202],
        [5, 213, 225, 389, 69, 260],
        [205, 124, 211, 6, 74, 183],
        [207, 207, 13, 1, 5, 29],
        [175, 196, 170, 270, 18, 178],
        [24, 10, 226, 63, 24, 259],
    ]
)

b = np.array([4, 56, 32, 47, 247, 28, 67, 218])
c = np.array([90, 302, 12, 25, 24, 87])

print("simplex", end="\n\n")

x1 = simplex(make_matrix(A, b, -c))

print("dual simplex", end="\n\n")

x2 = dual_simplex(make_dual_matrix(A, b, -c))

print(f"simplex: {x1[0]}\ndual simplex: {x2[0]}\ndelta: {np.abs(x1[0] - x2[0])}\n")
