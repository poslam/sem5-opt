import sys

import numpy as np
from tabulate import tabulate


def print_matrix(matrix):
    if len(matrix.shape) == 1:
        matrix = matrix.reshape((1, matrix.shape[0]))
    str_matrix = [[str(cell) for cell in row] for row in matrix]
    print(f"{tabulate(str_matrix, tablefmt='fancy_grid')}\n")


# Прямая задача
def simplex_method(c, A, b):

    # Число переменных и ограничений
    m, n = A.shape

    # Создаем симплекс-таблицу
    tableau = np.zeros((m + 1, n + m + 1))

    # Заполняем таблицу
    tableau[1:, :n] = A  # Ограничения
    tableau[1:, n : n + m] = np.eye(m)  # Добавляем базисные переменные
    tableau[1:, -1] = b  # Свободные члены
    tableau[0, :n] = -c  # Целевая функция (сверху)

    # Симплекс-итерации
    step = 0
    while True:
        print_matrix(tableau, header="Симплекс-таблица Шаг " + str(step + 1))

        # Шаг 1: Проверяем, достигнут ли оптимум
        if np.all(tableau[0, :-1] >= 0):
            print("Оптимум достигнут.")
            break

        # Шаг 2: Выбираем разрешающий столбец (самый отрицательный элемент в первой строке)
        pivot_col = np.argmin(tableau[0, :-1])
        print(f"Разрешающий столбец: {pivot_col  + 1}")

        # Шаг 3: Вычисляем отношение свободного члена к элементу столбца
        ratios = []
        for i in range(1, m + 1):
            if tableau[i, pivot_col] > 0:
                ratios.append(tableau[i, -1] / tableau[i, pivot_col])
            else:
                ratios.append(np.inf)
        pivot_row = (
            np.argmin(ratios) + 1
        )  # Сдвигаем индекс, т.к. первая строка - это `-c`

        if ratios[pivot_row - 1] == np.inf:
            raise ValueError("Задача не имеет ограниченного решения.")

        # Определяем разрешающий элемент
        pivot_element = tableau[pivot_row, pivot_col]
        print(f"Разрешающая строка: {pivot_row + 1}")
        print(f"Разрешающий элемент: {pivot_element}")

        # Шаг 4: Приводим разрешающий элемент к 1
        tableau[pivot_row, :] /= pivot_element

        # Шаг 5: Обновляем таблицу
        for i in range(m + 1):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

        step += 1

    # Оптимальное значение
    optimal_value = tableau[0, -1]

    # Вектор переменных
    x = np.zeros(n)
    for i in range(n):
        col = tableau[1:, i]
        if np.count_nonzero(col) == 1 and np.sum(col) == 1:
            x[i] = tableau[
                np.argmax(col) + 1, -1
            ]  # Учитываем смещение из-за строки `-c`

    return optimal_value, x


# Двойственная задача
def dual_simplex_method(A, b, c):

    # Транспонируем A, так как в двойственной задаче ограничения зависят от A^T
    A_T = A.T

    # Число переменных и ограничений
    m, n = A_T.shape

    # Создаем симплекс-таблицу
    tableau = np.zeros((m + 1, n + m + 1))

    # Заполняем таблицу
    tableau[:m, :n] = A_T  # Ограничения
    tableau[:m, n : n + m] = np.eye(m)  # Базисные переменные
    tableau[:m, -1] = c  # Вектор свободных членов
    tableau[-1, n : n + m] = -b  # Целевая функция

    # Симплекс-итерации
    step = 0
    while True:
        print_matrix(tableau, header="Симплекс-таблица Шаг " + str(step + 1))

        # Проверяем, достигнут ли оптимум
        if np.all(tableau[:-1, -1] >= 0):
            print("Оптимум достигнут.")
            break

        # Выбираем разрешающую строку (самый отрицательный элемент в последнем столбце)
        pivot_row = np.argmin(tableau[:-1, -1])
        print(f"Разрешающий столбец: {pivot_col  + 1}")

        # Выбираем разрешающий столбец
        pivot_col = None
        min_ratio = np.inf
        for j in range(n + m):
            if tableau[pivot_row, j] < 0:
                ratio = abs(tableau[-1, j] / tableau[pivot_row, j])
                if ratio < min_ratio:
                    min_ratio = ratio
                    pivot_col = j
        if pivot_col is None:
            raise ValueError("Задача не имеет ограниченного решения.")

        # Разрешающий элемент
        pivot_element = tableau[pivot_row, pivot_col]
        print(f"Разрешающая строка: {pivot_row + 1}")
        print(f"Разрешающий элемент: {pivot_element}")

        # Приводим разрешающий элемент к 1
        tableau[pivot_row, :] /= pivot_element

        # Обновляем таблицу
        for i in range(m + 1):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

        step += 1

    # Оптимальное значение
    optimal_value = tableau[-1, -1]

    # Переменные y
    y = np.zeros(m)
    for i in range(m):
        col = tableau[:m, n + i]
        if np.count_nonzero(col) == 1 and np.sum(col) == 1:
            y[i] = tableau[np.argmax(col), -1]

    return optimal_value, y


sys.stdout = open("lab3/output.txt", "w")

# A = generate_matrix((8, 6) , low=1 , high=100)
# b = generate_matrix((8, 1) , low=1 , high=100)
# c = generate_matrix((1, 6) , low=1 , high=100)

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

b = np.array([296, 85, 22, 47, 247, 28, 125, 218])

c = np.array([173, 299, 240, 120, 249, 86])

# print_matrix(A, header="Матрица A")

# print_matrix(b, header="Вектор b")

# print_matrix(c, header="Вектор c")


print("Прямая задача:")

opt_value, variables = simplex_method(c, A, b.T)
print("\nОптимальное значение:", opt_value)
print(variables)

print()

print()

print()


print("Двойственная задача:")

opt_value, variables = dual_simplex_method(A, b, c)
print("\nОптимальное значение:", opt_value)
print(variables)
