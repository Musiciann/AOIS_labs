from unittest import TestCase
import Binary_Operations
import pytest

class Test(TestCase):
    def test_decimal_to_binary_direct_code(self):
        binary_direct = Binary_Operations.decimal_to_binary_direct_code(2)
        assert binary_direct == "00000010"

    def test_decimal_to_binary_direct_code_zero(self):
        binary_direct = Binary_Operations.decimal_to_binary_direct_code(0)
        assert binary_direct == "0"

    def test_decimal_to_binary_direct_code_negative(self):
        binary_direct = Binary_Operations.decimal_to_binary_direct_code(-2)
        assert binary_direct == "10000010"

    def test_decimal_to_binary_inverse_code(self):
        binary_inverse = Binary_Operations.decimal_to_binary_inverse_code(2)
        assert binary_inverse == "00000010"

    def test_decimal_to_binary_additional_code(self):
        binary_additional = Binary_Operations.decimal_to_binary_additional_code(2)
        assert binary_additional == "00000010"

    def test_decimal_to_binary_inverse_code_negative(self):
        binary_inverse = Binary_Operations.decimal_to_binary_inverse_code(-2)
        assert binary_inverse == "11111101"

    def test_decimal_to_binary_additional_code_negative(self):
        binary_additional = Binary_Operations.decimal_to_binary_additional_code(-2)
        assert binary_additional == "11111110"

    def test_additional_code_binary_sum(self):
        binary_additional = Binary_Operations.additional_code_binary_sum(Binary_Operations.decimal_to_binary_additional_code(-2),
                                                                    Binary_Operations.decimal_to_binary_additional_code(2))
        assert binary_additional == "00000000"

    def test_binary_to_decimal_direct_code_negative(self):
        decimal = Binary_Operations.binary_to_decimal_direct_code('10000010')
        assert decimal == -2

    def test_binary_to_decimal_direct_code(self):
        decimal = Binary_Operations.binary_to_decimal_direct_code('00000010')
        assert decimal == 2

    def test_binary_to_decimal_additional_code(self):
        decimal = Binary_Operations.binary_to_decimal_additional_code(Binary_Operations.decimal_to_binary_additional_code(3))
        assert decimal == 3

    def test_binary_to_decimal_additional_code_negative(self):
        decimal = Binary_Operations.binary_to_decimal_additional_code(Binary_Operations.decimal_to_binary_additional_code(-3))
        assert decimal == -3

    def test_binary_multiply(self):
        binary_direct = Binary_Operations.binary_multiply('00001100','00000011')
        decimal = Binary_Operations.binary_to_decimal_direct_code(binary_direct)
        assert decimal == 36

    def test_binary_multiply_negative(self):
        binary_direct = Binary_Operations.binary_multiply('00001100','10000011')
        decimal = Binary_Operations.binary_to_decimal_direct_code(binary_direct)
        assert decimal == -36

    def test_binary_multiply_negatives(self):
        binary_direct = Binary_Operations.binary_multiply('10001100','10000011')
        decimal = Binary_Operations.binary_to_decimal_direct_code(binary_direct)
        assert decimal == 36

    def test_binary_division(self):
        binary_direct = Binary_Operations.binary_divide(Binary_Operations.decimal_to_binary_direct_code(10),
                                                      Binary_Operations.decimal_to_binary_direct_code(3))
        assert binary_direct == '011.01010101'

    def test_binary_division_zero_dividend(self):
        binary_direct = Binary_Operations.binary_divide('00000000', '00000010')
        assert binary_direct == '00'

    def test_binary_subtract(self):
        binary_num = Binary_Operations.binary_subtract('00000101', '00000010')
        assert binary_num == '11'

    def test_sum_floats_ieee754(self):
        result_float, result_binary = Binary_Operations.sum_floats_ieee754(22.5, 33.2)
        assert result_binary == '01000010010111101100110011001100'
        assert result_float == 55.69999694824219

