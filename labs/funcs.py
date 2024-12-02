import numpy as np
from tabulate import tabulate

ROUND_VAL = 3


def print_matrix(matrix: np.ndarray, header=None):
    matrix = matrix.round(ROUND_VAL)
    str_matrix = [[str(cell) for cell in row] for row in matrix]
    s = f"{tabulate(str_matrix, tablefmt='fancy_grid' ,)}\n"

    if header is not None:
        header = str(header).center(len(s.split("\n")[0]))
        print(header)

    print(s)
