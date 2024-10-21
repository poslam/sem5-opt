import os

import numpy as np
import pypandoc


def matrix_to_md(matrix):
    """Конвертирует numpy-массив в строку LaTeX для матрицы."""
    shape = matrix.shape
    matrix_str = r"\begin{bmatrix}"

    for i in range(shape[0]):
        row_str = " & ".join(map(str, matrix[i]))
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
