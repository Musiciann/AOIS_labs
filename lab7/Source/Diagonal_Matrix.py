import random
from constants import *

def print_matrix(matrix):
    for row in matrix:
        print(''.join(str(x) for x in row))

def generate_matrix(rows=16, cols=16, ones_probability=0.5):
    matrix = [[0 for _ in range(cols)] for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            if (i + j) % rows == i:
                matrix[i][j] = 1 if random.random() < ones_probability else 0
            else:
                matrix[i][j] = 1 if random.random() < ones_probability / 3 else 0

    for j in range(cols):
        if all(matrix[i][j] == 0 for i in range(rows)):
            i = random.randint(0, rows - 1)
            matrix[i][j] = 1

    return matrix

def logical_function_f6(x1, x2):
    return (not x1 and x2) or (x1 and not x2)

def logical_function_f9(x1, x2):
    return (x1 and x2) or (not x1 and not x2)

def logical_function_f4(x1, x2):
    return not x1 and x2

def logical_function_f11(x1, x2):
    return x1 or not x2

def apply_logical_function(matrix, rows, cols, word_index1, word_index2, target_index, func):
    word1 = [int(bit) for bit in extract_vertical_word(matrix, word_index1, rows, cols)]
    word2 = [int(bit) for bit in extract_vertical_word(matrix, word_index2, rows, cols)]

    result = []
    for i in range(rows):
        if func == 'f6':
            res_bit = logical_function_f6(word1[i], word2[i])
        elif func == 'f9':
            res_bit = logical_function_f9(word1[i], word2[i])
        elif func == 'f4':
            res_bit = logical_function_f4(word1[i], word2[i])
        elif func == 'f11':
            res_bit = logical_function_f11(word1[i], word2[i])
        else:
            res_bit = 0
        result.append(int(res_bit))

    word_result = result[(-1 * target_index):]

    for word_bit in result[:(-1 * target_index)]:
        word_result.append(word_bit)
    for i in range(rows):
        col_idx = target_index
        matrix[i][col_idx] = word_result[i]

def add_fields_with_key(matrix, key, rows, cols):
    key_bits = [int(bit) for bit in key]

    for word_idx in range(cols):
        v_bits = []
        for bit_pos in range(3):
            row = (word_idx + bit_pos) % rows
            v_bits.append(matrix[row][word_idx])

        if v_bits != key_bits:
            continue

        a_bits = []
        for bit_pos in range(3, 7):
            row = (word_idx + bit_pos) % rows
            a_bits.append(matrix[row][word_idx])

        b_bits = []
        for bit_pos in range(7, 11):
            row = (word_idx + bit_pos) % rows
            b_bits.append(matrix[row][word_idx])

        a_num = int(''.join(map(str, a_bits)), 2)
        b_num = int(''.join(map(str, b_bits)), 2)
        sum_result = a_num + b_num

        sum_bits = [int(bit) for bit in f"{sum_result:05b}"]
        for bit_pos in range(5):
            row = (word_idx + 11 + bit_pos) % rows
            matrix[row][word_idx] = sum_bits[bit_pos]

def get_diagonal_address(grid, col_num, height, width):
    diagonal_bits = []
    for bit_pos in range(width):
        current_row = (col_num + bit_pos) % height
        diagonal_bits.append(grid[current_row][bit_pos])
    return diagonal_bits

def extract_vertical_word(data_grid, word_pos, row_count, col_count):
    vertical_data = []
    for row in range(row_count):
        vertical_data.append(data_grid[row][word_pos])

    rotated_word = vertical_data[word_pos:] + vertical_data[:word_pos]
    return ''.join(map(str, rotated_word))

def sort_matrix_rows(grid, row_count, col_count, descending=True):
    remaining_rows = set(range(row_count))
    ordered_rows = []

    while remaining_rows:
        candidates = list(remaining_rows)
        selected = []

        for bit in range(col_count):
            target = 1 if descending else 0
            current_selection = []

            for row in candidates:
                if grid[row][bit] == target:
                    current_selection.append(row)

            if current_selection:
                candidates = current_selection
                if len(candidates) == 1:
                    break

        ordered_rows.extend(candidates)
        remaining_rows.difference_update(candidates)

    return ordered_rows
