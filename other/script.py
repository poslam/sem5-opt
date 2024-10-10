import os

import numpy as np
import pypandoc


def matrix_to_markdown(matrix):
    """Конвертирует numpy-массив в строку LaTeX для матрицы."""
    try:
        rows, cols = matrix.shape
    except:
        print(matrix)
        rows = 4
    matrix_str = r"\begin{bmatrix}"
    for i in range(rows):
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
            + matrix_to_markdown(matrix)
            + r"""
$$
"""
            for matrix in matrices
        ]
    )


# Пример матрицы
A = np.array(
    [
        [26.20056955320149, -58.76847438658791, 414.685599118923, 254.48571079046798],
        [-58.76847438658791, 945.449926413821, 355.0397007546424, 487.73320892614504],
        [414.685599118923, 355.0397007546424, -604.0485273715911, -423.5778583766898],
        [254.48571079046798, 487.73320892614504, -423.5778583766898, 257.4102340464],
    ]
)
b = np.array([1, 2, 3, 4]).reshape(-1, 1)
x0 = np.array([1, 1, 1, 1]).reshape(-1, 1)

matrices = [A]

# Генерация Markdown-контента
markdown_content = (
    r"""
"""
    + matrices_to_md(matrices)
    + r"""
"""
)

output_file = "other/matrix.docx"

pypandoc.convert_text(
    markdown_content,
    "docx",
    format="md",
    outputfile=output_file,
)

os.system(f"open {output_file}")
