# src: https://drive.google.com/file/d/1-ItLEAsiiBhwEhRflFU0Te9w-D73Boe5/view?usp=drive_link
# О.В. Болотникова, Д.В. Тарасов: "Линейное программирование: симплекс-метод и двойственность"
# pages: 28, 52

"""
for max:
index_of_element = argmax()
... <= 0: break
return -
c
"""

import numpy as np

from labs.funcs import print_matrix, print_matrix_latex

ROUND_VAL = 3


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
            np.hstack((np.reshape(c, (A.shape[0], 1)), -A, np.eye(A.shape[0]))),
            np.hstack(((np.array([0])), -b, np.zeros((A.shape[0])))),
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
                f"index: {(min_line, int(index_of_element))}\n"
                + f"focus func val: {round(simplex_matrix[-1, int(index_of_element)], ROUND_VAL)}\n"
                + f"focus val: {round(simplex_matrix[min_line, int(index_of_element)], ROUND_VAL)}",
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

    print("result: ")
    print_matrix(simplex_matrix)

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
                f"index: {(int(index_of_element), min_column)}\n"
                + f"focus func val: {round(simplex_matrix[:-1, 0][index_of_element], ROUND_VAL)}\n"
                + f"focus val: {round(simplex_matrix[int(index_of_element), min_column], ROUND_VAL)}",
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

    print("result: ")
    print_matrix(simplex_matrix)

    return simplex_matrix[-1, 0], simplex_matrix


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
print("Верхняя цена игры:", min(tmp))

beta = A.min()

A_cap: np.ndarray = A + np.abs(beta)

print_matrix(A_cap)

# b = np.array([296, 85, 22, 47, 247, 28, 125, 218])
# c = np.array([173, 299, 240, 120, 249, 86])

# print("simplex", end="\n\n")

# x1 = simplex(make_matrix(A, b, -c))

# print("dual simplex", end="\n\n")

# x2 = dual_simplex(make_dual_matrix(A.T, b, -c))

# print(f"simplex: {x1[0]}\ndual simplex: {x2[0]}\ndelta: {np.abs(x1[0] - x2[0])}\n")
