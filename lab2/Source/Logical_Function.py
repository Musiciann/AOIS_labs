import itertools
from constants import *

class LogicalFunction:
    def __init__(self, expression):
        self.expression = expression.replace(' ', '')
        self.variables = sorted(set(filter(lambda x: x in VARIABLES, self.expression)))
        self.num_vars = len(self.variables)
        self.truth_table = []

    def binary_to_decimal_direct_code(self, binary_num: str) -> tuple:
        decimal_value = 0
        sign = binary_num[0]
        binary_value = binary_num
        for i, digit in enumerate(reversed(binary_num[1:])):
            decimal_value += int(digit) * (2 ** i)

        if sign == '1':
            decimal_value *= -1

        return binary_value, decimal_value

    def binary_result_decimal(self, result: tuple) -> int:
        result_string = ""
        for num in result:
            result_string += str(num)
        result = "".join(result_string)
        decimal_result = self.binary_to_decimal_direct_code(result.zfill(8))
        return decimal_result[1]

    def evaluate(self, values) -> int:
        return self.evaluate_expr(self.expression, values)

    def evaluate_expr(self, expr, values) -> int:
        current_result = dict(zip(self.variables, values))
        return self.evaluate_recursive(expr, current_result)

    def evaluate_recursive(self, expr, current_result) -> int:
        stack = []
        i = 0
        while i < len(expr):
            if expr[i] in current_result:
                stack.append(current_result[expr[i]])
            elif expr[i] == '!':
                i += 1
                i, result = self.handle_not(expr, i, current_result)
                stack.append(result)
            elif expr[i] == '&':
                i, result = self.handle_and(expr, i, stack, current_result)
                stack.append(result)
            elif expr[i] == '|':
                i, result = self.handle_or(expr, i, stack, current_result)
                stack.append(result)
            elif expr[i] == '~':
                i, result = self.handle_equivalence(expr, i, stack, current_result)
                stack.append(result)
            elif expr[i] == '>':
                i, result = self.handle_implication(expr, i, stack, current_result)
                stack.append(result)
            elif expr[i] == '(':
                j = self.find_matching_parenthesis(expr, i)
                inner_value = self.evaluate_recursive(expr[i + 1:j], current_result)
                stack.append(inner_value)
                i = j
            i += 1

        total_result = stack[0]
        for value in stack[1:]:
            total_result = total_result & value
        return total_result

    def handle_not(self, expr, i, current_result):
        if expr[i] == '(':
            j = self.find_matching_parenthesis(expr, i)
            inner_value = self.evaluate_recursive(expr[i + 1:j], current_result)
            return j, 1 - inner_value
        else:
            return i, 1 - current_result[expr[i]]

    def handle_and(self, expr, i, stack, current_result):
        if stack:
            a = stack.pop()
            i += 1
            if expr[i] == '(':
                j = self.find_matching_parenthesis(expr, i)
                inner_value = self.evaluate_recursive(expr[i + 1:j], current_result)
                return j, (a * inner_value)
            elif expr[i] == '!':
                i += 1
                if expr[i] == '(':
                    j = self.find_matching_parenthesis(expr, i)
                    inner_value = self.evaluate_recursive(expr[i + 1:j], current_result)
                    return j, (a * (1 - inner_value))
                else:
                    return i, (a * (1 - current_result[expr[i]]))
            else:
                b = current_result[expr[i]]
                return i, (a * b)
        return i, 0

    def handle_or(self, expr, i, stack, current_result):
        if stack:
            a = stack.pop()
            i += 1
            if expr[i] == '(':
                j = self.find_matching_parenthesis(expr, i)
                inner_value = self.evaluate_recursive(expr[i + 1:j], current_result)
                return j, max(a, inner_value)
            elif expr[i] == '!':
                i += 1
                if expr[i] == '(':
                    j = self.find_matching_parenthesis(expr, i)
                    inner_value = self.evaluate_recursive(expr[i + 1:j], current_result)
                    return j, max(a, (1 - inner_value))
                else:
                    return i, max(a, (1 - current_result[expr[i]]))
            else:
                b = current_result[expr[i]]
                return i, max(a, b)
        return i, 0

    def handle_equivalence(self, expr, i, stack, current_result):
        if stack:
            a = stack.pop()
            i += 1
            if expr[i] == '(':
                j = self.find_matching_parenthesis(expr, i)
                inner_value = self.evaluate_recursive(expr[i + 1:j], current_result)
                return j, int(a == inner_value)
            elif expr[i] == '!':
                i += 1
                if expr[i] == '(':
                    j = self.find_matching_parenthesis(expr, i)
                    inner_value = self.evaluate_recursive(expr[i + 1:j], current_result)
                    return j, int(a == (1 - inner_value))
                else:
                    return i, int(a == (1 - current_result[expr[i]]))
            else:
                b = current_result[expr[i]]
                return i, int(a == b)
        return i, 0

    def handle_implication(self, expr, i, stack, current_result):
        if stack:
            a = stack.pop()
            i += 1
            if expr[i] == '(':
                j = self.find_matching_parenthesis(expr, i)
                inner_value = self.evaluate_recursive(expr[i + 1:j], current_result)
                return j, int(not a or inner_value)
            elif expr[i] == '!':
                i += 1
                if expr[i] == '(':
                    j = self.find_matching_parenthesis(expr, i)
                    inner_value = self.evaluate_recursive(expr[i + 1:j], current_result)
                    return j, int(not a or (1 - inner_value))
                else:
                    return i, int(not a or (1 - current_result[expr[i]]))
            else:
                b = current_result[expr[i]]
                return i, int(not a or b)
        return i, 0

    def find_matching_parenthesis(self, expr, start) -> int:
        count = 1
        for i in range(start + 1, len(expr)):
            if expr[i] == '(':
                count += 1
            elif expr[i] == ')':
                count -= 1
                if count == 0:
                    return i
        raise ValueError("Mismatched parentheses in expression")

    def truth_table_result(self) -> tuple:
        binary_result = []
        binary_result_sign = [0]

        for i in range(len(self.truth_table)):
            binary_result.append(self.truth_table[i][1])
            binary_result_sign.append(self.truth_table[i][1])

        decimal_result = self.binary_result_decimal(tuple(binary_result_sign))

        return binary_result, decimal_result

    def generate_truth_table(self) -> tuple:
        for value in itertools.product([0, 1], repeat=self.num_vars):
            result = self.evaluate(value)
            self.truth_table.append((value, result))

        return self.truth_table, self.variables, self.expression

    def generate_sdnf(self) -> str:
        sdnf_terms = []

        for (values, result) in self.truth_table:
            if result:
                term = []
                for j in range(self.num_vars):
                    if values[j] == 1:
                        term.append(self.variables[j])
                    else:
                        term.append(f'!{self.variables[j]}')
                sdnf_terms.append(f'({" & ".join(term)})')

        return ' | '.join(sdnf_terms)

    def generate_sdnf_numeric(self) -> list:
        sdnf_terms_numeric = []

        for (values, result) in self.truth_table:
            if result:
                numbers = []
                binary_number = ''
                for value in values:
                    binary_number += str(value)
                decimal = self.binary_to_decimal_direct_code(binary_number.zfill(8))
                sdnf_terms_numeric.append(decimal[1])

        return sdnf_terms_numeric

    def generate_scnf(self) -> str:
        sknf_terms = []

        for (values, result) in self.truth_table:
            if not result:
                term = []
                for j in range(self.num_vars):
                    if values[j] == 0:
                        term.append(self.variables[j])
                    else:
                        term.append(f'!{self.variables[j]}')
                sknf_terms.append(f'({" | ".join(term)})')

        return ' & '.join(sknf_terms)

    def generate_scnf_numeric(self) -> list:
        scnf_terms_numeric = []
        for (values, result) in self.truth_table:
            if not result:
                numbers = []
                binary_number = ''
                for value in values:
                    binary_number += str(value)
                decimal = self.binary_to_decimal_direct_code(binary_number.zfill(8))
                scnf_terms_numeric.append(decimal[1])

        return scnf_terms_numeric