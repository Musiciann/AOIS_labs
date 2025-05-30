from unittest import TestCase
from Diagonal_Matrix import *

class Test(TestCase):
    def test_extract_vertical_word(self):
        test_matrix = [
                [1, 0, 0, 1, 1],
                [0, 0, 0, 1, 0],
                [1, 1, 0, 1, 0],
                [1, 0, 0, 1, 0],
                [1, 0, 1, 1, 1]
                  ]
        test_word = extract_vertical_word(test_matrix, 4, 5, 5)
        assert test_word == '11000'

    def test_get_diagonal_address(self):
        test_matrix = [
                [1, 0, 0, 1, 1],
                [0, 0, 0, 1, 0],
                [1, 1, 0, 1, 0],
                [1, 0, 0, 1, 0],
                [1, 0, 1, 1, 1]
                  ]
        test_address = get_diagonal_address(test_matrix, 3, 5, 5)
        assert test_address == [1, 0, 0, 1, 0]

    def test_logical_function_f6(self):
        test_matrix = generate_matrix(5, 5)
        test_matrix = [
                [1, 0, 0, 1, 1],
                [0, 0, 0, 1, 0],
                [1, 1, 0, 1, 0],
                [1, 0, 0, 1, 0],
                [1, 0, 1, 1, 1]
                  ]
        apply_logical_function(test_matrix, 5, 5, 0, 1, 2, 'f6')
        test_word = extract_vertical_word(test_matrix, 2, 5, 5)
        assert test_word == '11111'

    def test_logical_function_f4(self):
        test_matrix = [
                [1, 0, 0, 1, 1],
                [0, 0, 0, 1, 0],
                [1, 1, 0, 1, 0],
                [1, 0, 0, 1, 0],
                [1, 0, 1, 1, 1]
                  ]
        apply_logical_function(test_matrix, 5, 5, 0, 1, 2, 'f4')
        test_word = extract_vertical_word(test_matrix, 2, 5, 5)
        assert test_word == '01000'

    def test_logical_function_f9(self):
        test_matrix = [
                [1, 0, 0, 1, 1],
                [0, 0, 0, 1, 0],
                [1, 1, 0, 1, 0],
                [1, 0, 0, 1, 0],
                [1, 0, 1, 1, 1]
                  ]
        apply_logical_function(test_matrix, 5, 5, 1, 2, 2, 'f9')
        test_word = extract_vertical_word(test_matrix, 2, 5, 5)
        assert test_word == '10011'

    def test_logical_function_f11(self):
        test_matrix = [
                [1, 0, 0, 1, 1],
                [0, 0, 0, 1, 0],
                [1, 1, 0, 1, 0],
                [1, 0, 0, 1, 0],
                [1, 0, 1, 1, 1]
                  ]
        apply_logical_function(test_matrix, 5, 5, 0, 1, 2, 'f11')
        test_word = extract_vertical_word(test_matrix, 2, 5, 5)
        assert test_word == '10111'

    def test_add_fields_with_key(self):
        test_matrix = [
                [1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1],
                [1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
                [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1],
                [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1],
                [0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1],
                [1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
                [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0],
                [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
                [1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1],
                [0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
                [1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1],
                [1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1],
                  ]
        add_fields_with_key(test_matrix, "111", ROWS, COLS)
        test_word = extract_vertical_word(test_matrix, 0, 16, 16)
        assert test_word == '1111110110111011'

    def test_sort_matrix_rows(self):
        test_matrix = [
            [1, 0, 0, 1, 1],
            [0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 1, 1, 1]
        ]
        sorted_matrix = sort_matrix_rows(test_matrix, 5, 5, False)
        assert sorted_matrix == [1, 3, 0, 4, 2]

