from unittest import TestCase
from LF import LogicalFunction


class Test(TestCase):
    def test_binary_to_decimal_direct_code(self):
        expr = "(a | b) & !c"
        logical_func = LogicalFunction(expr)
        binary_num, decimal_num = logical_func.binary_to_decimal_direct_code('10000010')
        assert binary_num == '10000010'
        assert decimal_num == -2

    def test_binary_result_decimal(self):
        expr = "(a | b) & !c"
        logical_func = LogicalFunction(expr)
        decimal_result = logical_func.binary_result_decimal((1, 0, 0, 0, 0, 0, 1, 0))
        assert decimal_result == -2

    def test_evaluate_expr(self):
        expr = "!(a&b)|!c~(d>(n|(g&(k|m))))"
        logical_func = LogicalFunction(expr)
        result = logical_func.evaluate((0, 0, 0, 0, 0, 0, 0, 0))
        assert result == 1

    def test_truth_table_result(self):
        expr = "(a | b) & !c"
        logical_func = LogicalFunction(expr)
        logical_func.generate_truth_table()
        binary_result, decimal_result = logical_func.truth_table_result()
        assert binary_result == [0, 0, 1, 0, 1, 0, 1, 0]
        assert decimal_result == 42

    def test_generate_sdnf(self):
        expr = "(a | b) & !c"
        logical_func = LogicalFunction(expr)
        logical_func.generate_truth_table()
        sdnf = logical_func.generate_sdnf()
        assert sdnf == '(!a & b & !c) | (a & !b & !c) | (a & b & !c)'

    def test_generate_sknf(self):
        expr = "(a | b) & !c"
        logical_func = LogicalFunction(expr)
        logical_func.generate_truth_table()
        scnf = logical_func.generate_scnf()
        assert scnf == '(a | b | c) & (a | b | !c) & (a | !b | !c) & (!a | b | !c) & (!a | !b | !c)'

    def test_generate_sdnf_numeric(self):
        expr = "(a | b) & !c"
        logical_func = LogicalFunction(expr)
        logical_func.generate_truth_table()
        sdnf = logical_func.generate_sdnf_numeric()
        assert sdnf == [2, 4, 6]

    def test_generate_scnf_numeric(self):
        expr = "(a | b) & !c"
        logical_func = LogicalFunction(expr)
        logical_func.generate_truth_table()
        scnf = logical_func.generate_scnf_numeric()
        assert scnf == [0, 1, 3, 5, 7]
