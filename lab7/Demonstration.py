from Diagonal_Matrix import *
from constants import *

if __name__ == "__main__":

    matrix = generate_matrix(ROWS, COLS)

    print("Демонстрация общего вида исходной матрицы (16x16):")
    print_matrix(matrix)

    print("\nДемонстрация считывания адресного столбца (индекс: 2):")
    bits = get_diagonal_address(matrix, 2, ROWS, COLS)
    print(bits)

    print("\nДемонстрация считывания слова (индекс: 3):")
    word = extract_vertical_word(matrix, 3, ROWS, COLS)
    print(word)

    print("\nДемонстрация f6: применение функции f6 (Неравнозначность (ИЛИ - ИЛИ)) к словам (индексы) 2 и 8, результат записать в слово 15:")
    apply_logical_function(matrix, ROWS, COLS, 2, 8, 15, 'f6')
    print("Матрица после операции:")
    print_matrix(matrix)

    print("\nДемонстрация f11: применение функции f11 (Импликация от 2-го аргумента к первому (НЕТ - НЕ)) к словам (индексы) 1 и 10, результат записать в слово 0:")
    apply_logical_function(matrix, ROWS, COLS, 1, 10, 0, 'f11')
    print("Матрица после операции:")
    print_matrix(matrix)

    print("\nДемонстрация f9: применение функции f9 (Эквивалентность (И - И)) к словам (индексы) 4 и 7, результат записать в слово 14:")
    apply_logical_function(matrix, ROWS, COLS, 4, 7, 14, 'f9')
    print("Матрица после операции:")
    print_matrix(matrix)

    print("\nДемонстрация f4: применение функции f4 (Запрет 2-го аргумента (НЕТ)) к словам (индексы) 2 и 13, результат записать в слово 1:")
    apply_logical_function(matrix, ROWS, COLS, 2, 13, 1, 'f4')
    print("Матрица после операции:")
    print_matrix(matrix)

    print("\nДемонстрация сложения полей A и B для слов с V = 100:")
    add_fields_with_key(matrix, "100", ROWS, COLS)
    print("Матрица после операции:")
    print_matrix(matrix)

    print("\nДемонстрация сортировки: упорядоченная выборка (по убыванию):")
    sorted_indices = sort_matrix_rows(matrix, ROWS, COLS, descending=True)
    sorted_matrix = [matrix[i] for i in sorted_indices]
    print_matrix(sorted_matrix)

    print("\nДемонстрация сортировки: упорядоченная выборка (по убыванию):")
    sorted_indices = sort_matrix_rows(matrix, ROWS, COLS, descending=False)
    sorted_matrix = [matrix[i] for i in sorted_indices]
    print_matrix(sorted_matrix)