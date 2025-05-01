from unittest import TestCase

from Minimization import LogicalFunctionMinimization

class TestLogicalFunctionMinimization(TestCase):
    def test_minimize_sdnf_quine(self):
        lf = LogicalFunctionMinimization("(a&b)|!c")
        lf.generate_truth_table()
        result = lf.minimize_sdnf_quine()
        assert result[-14:] == '(!c) | (a & b)' or result[-14:] == '(a & b) | (!c)'

    def test_minimize_sknf_quine(self):
        lf = LogicalFunctionMinimization("(a&b)|!c")
        lf.generate_truth_table()
        result = lf.minimize_sknf_quine()
        assert result[-19:] == '(b | !c) & (a | !c)' or result[-19:] == '(a | !c) & (b | !c)'

    def test_minimize_sdnf_calculational(self):
        lf = LogicalFunctionMinimization("(a&b)|!c")
        lf.generate_truth_table()
        result = lf.minimize_sdnf_calculational()
        assert result[-10:] == '!c v a & b' or result[-10:] == 'a & b v !c'

    def test_minimize_sknf_calculational(self):
        lf = LogicalFunctionMinimization("(a&b)|!c")
        lf.generate_truth_table()
        result = lf.minimize_sknf_calculational()
        assert result[-15:] == 'b | !c & a | !c' or result[-15:] == 'a | !c & b | !c'

    def test_minimize_sdnf_karnaugh(self):
        lf = LogicalFunctionMinimization("!(!a>!b)|c")
        lf.generate_truth_table()
        result = lf.minimize_sdnf_karnaugh()
        assert result[-14:] == '(c) ∨ (!a & b)'

    def test_minimize_sknf_karnaugh(self):
        lf = LogicalFunctionMinimization("!(!a>!b)|c")
        lf.generate_truth_table()
        result = lf.minimize_sknf_karnaugh()
        assert result[-18:] == '(b | c) & (!a | c)'