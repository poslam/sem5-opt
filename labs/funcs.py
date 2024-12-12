import numpy as np
from tabulate import tabulate

ROUND_VAL = 3


def print_matrix(matrix: np.ndarray, header: str = None):
    if len(matrix.shape) == 1:
        matrix = matrix.reshape((1, matrix.shape[0]))

    matrix = matrix.round(ROUND_VAL)
    str_matrix = [[str(cell) for cell in row] for row in matrix]
    s = f"{tabulate(str_matrix, tablefmt='fancy_grid' ,)}\n"

    if header is not None:
        header = str(header).center(len(s.split("\n")[0]))
        print(header)

    print(s)


def print_matrix_latex(matrix: np.ndarray, header: str = None):
    if len(matrix.shape) == 1:
        matrix = matrix.reshape((1, matrix.shape[0]))

    matrix = matrix.round(ROUND_VAL)
    str_matrix = [[str(cell) for cell in row] for row in matrix]
    s = (
        "$$ \\begin{bmatrix} "
        + " \\\\ ".join([" & ".join(row) for row in str_matrix])
        + " \\end{bmatrix} $$"
    )
    print(s)
