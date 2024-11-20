import sys

import numpy as np

from labs.funcs import *


# Прямая задача
def simplex_method(tableau: np.ndarray):
    m, n = tableau.shape
    step = 0

    while np.any(tableau[0, :-1] < 0):
        step += 1
        print(f"Шаг {step}:\n")

        pivot_col = -1
        min_value = 0
        for j in range(
            n - 1
        ):  # Все элементы строки целевой функции, кроме последнего столбца (столбца b)
            if (
                tableau[0, j] < min_value
            ):  # Ищем минимальное по значению (отрицательное)
                min_value = tableau[0, j]
                pivot_col = j

        # Шаг 3: Выбираем выходящую переменную — наименьшее положительное отношение свободного члена к элементу столбца
        pivot_row = -1
        min_ratio = np.float64("inf")
        for i in range(
            1, m
        ):  # Начинаем с 1-й строки, т.к. 0-я строка — это строка целевой функции
            if tableau[i, pivot_col] > 0:  # Ищем положительные элементы в столбце
                ratio = tableau[i, -1] / tableau[i, pivot_col]
                if (
                    ratio < min_ratio and ratio >= 0
                ):  # Ищем минимальное положительное отношение
                    min_ratio = ratio
                    pivot_row = i

        # Если нет подходящих строк, задача неограничена
        if pivot_row == -1:
            raise ValueError(
                "Задача неограничена (нет ограничений, которые сдерживают решение)."
            )

        # Шаг 4: Поворот — обновляем таблицу симплекс-метода
        pivot_value = tableau[pivot_row, pivot_col]

        # Преобразуем разрешающую строку (делим её на разрешающий элемент)
        tableau[pivot_row] /= pivot_value

        # Преобразуем остальные строки (вычитаем из них соответствующие пропорциональные значения)
        for i in range(m):
            if i != pivot_row:
                tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]

        # Вывод информации о текущем шаге

        print()
        print(
            f"Разрешающий элемент: строка {pivot_row + 1}, столбец {pivot_col + 1} = {pivot_value}"
        )
        print()

        print_matrix(tableau, header="Текущая симплекс таблица:")

        print("═" * 500)

    solution = np.zeros(n - 1)  # Все переменные по умолчанию равны 0
    for i in range(
        1, m
    ):  # Пропускаем первую строку (целевая функция) # по ширине матрицы А
        # Если в строке разрешающий элемент на диагонали

        pivot_col = np.where(tableau[i, :-1] == 1)[0]  # Ищем индекс переменной в строке
        if pivot_col.size > 0:
            solution[pivot_col[0]] = tableau[i, -1]  # Считываем значение переменной

    # Если все элементы целевой функции неотрицательные, то решение найдено
    return tableau[0, -1], solution  # Решение — это столбец b (все переменные решения)


def main():
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

    ### прямая задача ---------------------------------------------------------------

    print("Прямая задача:")

    m, n = A.shape

    tableau = np.zeros((m + 1, n + m + 1))

    tableau[1:, :n] = A  # Ограничения
    tableau[1:, n : n + m] = np.eye(m)  # Добавляем базисные переменные
    tableau[1:, -1:] = b  # Свободные члены
    tableau[0, :n] = -c  # Целевая функция (сверху)

    print_matrix(tableau, header="Исходная таблица")

    x, result = simplex_method(tableau)
    result = result[: A.shape[1]]

    print_matrix(np.array([result]), header="Начальное угловое решение")
    print(f"Целевая функция: {x}")
    print("\n\n", "═" * 500, "\n", "═" * 500, "\n", "═" * 500, "\n\n")

    ### вспомогательная задача ----------------------------------------------------------

    print("Вспомогательная задача")

    A_T = A.T
    b_T = b.T
    c_T = c.T

    m, n = A_T.shape

    tableau = np.zeros((m + 1, n + m + m + 1))
    tableau[1:, :n] = A_T  # Ограничения
    tableau[1:, n : n + m] = -1 * np.eye(m)  # Добавляем базисные переменные
    tableau[1:, n + m : n + m + m] = np.eye(m)  # Добавляем базисные переменные
    tableau[1:, -1:] = c_T  # Свободные члены
    tableau[0, n * 2 - 2 : -1] = np.array(
        [1 for i in range(m)]
    )  # Целевая функция (сверху)

    print_matrix(tableau, header="Исходная таблица")

    for i in range(1, m + 1):
        tableau[0] += tableau[i] * -1

    x, result = simplex_method(tableau)

    print_matrix(np.array([result]), header="Начальное угловое решение")
    print(f"Целевая функция: {x}")
    print("\n\n", "═" * 500, "\n", "═" * 500, "\n", "═" * 500, "\n\n")

    ### двойственная задача ----------------------------------------------------------

    print("Двойственная задача")

    # tableau = np.hstack((tableau[: , :n  ] ))

    tableau = np.hstack((tableau[:, : n + m], tableau[:, -1:]))
    filler_b = np.array([[0 for _ in range(tableau.shape[1] - b_T.shape[1])]])
    filler_b = np.hstack((b_T, filler_b))

    tableau[0] = filler_b

    print_matrix(tableau, header="Исходная таблица")

    x, result = simplex_method(tableau)

    print_matrix(np.array([result]), header="Решение")
    print(f"Целевая функция: {x}")


if __name__ == "__main__":
    sys.stdout = open("./labs/output.txt", "w")
    main()
