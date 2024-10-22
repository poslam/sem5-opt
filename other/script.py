import os

import numpy as np
import pypandoc


def matrix_to_md(matrix):
    shape = matrix.shape
    if len(shape) == 1:
        shape = (shape[0], 1)

    matrix_str = r"\begin{bmatrix}"
    for i in range(shape[1]):
        row_str = " & ".join([str(x) for x in list(matrix)])
        matrix_str += row_str + r" \\ "  # Используем \\ для перехода на новую строку
    matrix_str = matrix_str[:-3]  # Убираем последний перевод строки
    matrix_str += r"\end{bmatrix}"
    return matrix_str


def matrices_to_md(matrices):
    return "\n".join(
        [
            r"""
$$
"""
            + matrix_to_md(matrix)
            + r"""
$$
"""
            for matrix in matrices
        ]
    )


def matrices_to_word(matrices, open_file=False):
    if not isinstance(matrices, list):
        matrices = [matrices]

    markdown_content = matrices_to_md(matrices)

    output_file = "other/matrix.docx"

    pypandoc.convert_text(
        markdown_content,
        "docx",
        format="md",
        outputfile=output_file,
    )

    if open_file:
        os.system(f"open {output_file}")
