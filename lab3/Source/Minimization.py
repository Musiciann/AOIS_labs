from collections import defaultdict
from LF import LogicalFunction
import math
import re

VARIABLES = 'abcdefghijklmnopqrstuvwxyz'

class LogicalFunctionMinimization(LogicalFunction):

    def combine_terms(self, term1, term2):
        if len(term1) != len(term2):
            return None
        diff = 0
        combined = []
        for c1, c2 in zip(term1, term2):
            if c1 != c2:
                if c1 == 'X' or c2 == 'X':
                    return None
                diff += 1
                combined.append('X')
            else:
                combined.append(c1)
        return ''.join(combined) if diff == 1 else None

    def get_binary_terms(self, is_sdnf=True):
        terms = []
        for (values, result) in self.truth_table:
            if (is_sdnf and result) or (not is_sdnf and not result):
                binary = ''.join(map(str, values))
                terms.append(binary)
        return terms

    def covers(self, implicant, term):
        for i in range(len(implicant)):
            imp_bit = implicant[i]
            term_bit = term[i]
            if imp_bit == 'X':
                continue
            if imp_bit != term_bit:
                return False
        return True

    def minimize_sdnf_calculational(self):
        binary_terms = self.get_binary_terms(is_sdnf=True)
        stages, prime_implicants = self._compute_combining_stages(binary_terms)
        coverage = self._build_coverage(binary_terms, prime_implicants)
        essential = self._find_essential_implicants(coverage)
        minimal_implicants = self._find_minimal_implicants(essential, binary_terms, prime_implicants)
        return self._generate_output(stages, prime_implicants, minimal_implicants)

    def _compute_combining_stages(self, binary_terms):
        stages = []
        current_terms = binary_terms.copy()
        prime_implicants = []

        while True:
            new_terms, stage_info, used = self._process_single_stage(current_terms)
            prime_implicants = self._update_prime_implicants(current_terms, used, prime_implicants)

            if not stage_info:
                break

            stages.append({
                'stage_info': stage_info,
                'new_terms': new_terms.copy()
            })
            current_terms = new_terms

        prime_implicants = list(set(prime_implicants))
        return stages, prime_implicants

    def _process_single_stage(self, current_terms):
        new_terms = []
        used = set()
        stage_info = []

        for i in range(len(current_terms)):
            for j in range(i + 1, len(current_terms)):
                combined = self.combine_terms(current_terms[i], current_terms[j])
                if combined:
                    used.add(i)
                    used.add(j)
                    if combined not in new_terms:
                        new_terms.append(combined)
                        expr1 = self.binary_term_to_expr(current_terms[i])
                        expr2 = self.binary_term_to_expr(current_terms[j])
                        combined_expr = self.binary_term_to_expr(combined)
                        stage_info.append(f"{expr1} ∨ {expr2} => {combined_expr}")

        return new_terms, stage_info, used

    def _update_prime_implicants(self, current_terms, used, prime_implicants):
        for idx in range(len(current_terms)):
            if idx not in used and current_terms[idx] not in prime_implicants:
                prime_implicants.append(current_terms[idx])
        return prime_implicants

    def _build_coverage(self, binary_terms, prime_implicants):
        coverage = {t: [] for t in binary_terms}
        for imp in prime_implicants:
            for term in binary_terms:
                if self.covers(imp, term):
                    coverage[term].append(imp)
        return coverage

    def _find_essential_implicants(self, coverage):
        essential = []
        for term, imps in coverage.items():
            if len(imps) == 1:
                essential.append(imps[0])
        return list(set(essential))

    def _find_minimal_implicants(self, essential, binary_terms, prime_implicants):
        minimal_implicants = essential.copy()
        remaining = [t for t in binary_terms if not any(self.covers(imp, t) for imp in minimal_implicants)]

        for imp in prime_implicants:
            if imp in minimal_implicants:
                continue
            for term in remaining:
                if self.covers(imp, term):
                    minimal_implicants.append(imp)
                    remaining = [t for t in remaining if not self.covers(imp, t)]
                    break
        return minimal_implicants

    def _generate_output(self, stages, prime_implicants, minimal_implicants):
        stage_output = ["Этапы склеивания:"]
        for stage in stages:
            stage_output.extend(stage['stage_info'])
            stage_output.append("Результат: " + " v ".join([self.binary_term_to_expr(t) for t in stage['new_terms']]))
            stage_output.append(
                "Простые импликанты: " + " v ".join([self.binary_term_to_expr(t) for t in prime_implicants]))
            stage_output.append(
                "Минимальная СДНФ: " + " v ".join([self.binary_term_to_expr(t) for t in minimal_implicants]))
        return "\n".join(stage_output)

    def minimize_sknf_calculational(self):
        binary_terms = self.get_binary_terms(is_sdnf=False)
        stages = []
        prime_implicants = []
        current_terms = binary_terms.copy()

        current_terms, stages, prime_implicants = self.process_combining_stages(
            binary_terms, current_terms, stages, prime_implicants
        )

        coverage = self.build_coverage(binary_terms, prime_implicants)
        essential = self.find_essential_implicants(coverage)
        minimal_implicants = self.find_minimal_implicants(
            binary_terms, essential, prime_implicants, coverage
        )

        return self.format_output(stages, prime_implicants, minimal_implicants)

    def process_combining_stages(self, binary_terms, current_terms, stages, prime_implicants):
        while True:
            new_terms, stage_info, used = self.combine_terms_in_stage(current_terms)

            self.collect_unused_terms(current_terms, used, prime_implicants)

            if not stage_info:
                break

            stages.append({
                'stage_info': stage_info,
                'new_terms': new_terms.copy()
            })
            current_terms = new_terms

        prime_implicants.extend(current_terms)
        prime_implicants = list(set(prime_implicants))
        return current_terms, stages, prime_implicants

    def combine_terms_in_stage(self, current_terms):
        new_terms = []
        used = set()
        stage_info = []

        for i in range(len(current_terms)):
            for j in range(i + 1, len(current_terms)):
                combined = self.combine_terms(current_terms[i], current_terms[j])
                if combined:
                    used.update({i, j})
                    if combined not in new_terms:
                        new_terms.append(combined)
                        expr1 = self.binary_term_to_expr(current_terms[i], False)
                        expr2 = self.binary_term_to_expr(current_terms[j], False)
                        combined_expr = self.binary_term_to_expr(combined, False)
                        stage_info.append(f"({expr1}) & ({expr2}) => ({combined_expr})")

        return new_terms, stage_info, used

    def collect_unused_terms(self, current_terms, used, prime_implicants):
        for idx, term in enumerate(current_terms):
            if idx not in used and term not in prime_implicants:
                prime_implicants.append(term)

    def build_coverage(self, binary_terms, prime_implicants):
        coverage = defaultdict(list)
        for term in binary_terms:
            for imp in prime_implicants:
                if self.covers(imp, term):
                    coverage[term].append(imp)
        return coverage

    def find_essential_implicants(self, coverage):
        essential = []
        for imps in coverage.values():
            if len(imps) == 1:
                essential.append(imps[0])
        return list(set(essential))

    def find_minimal_implicants(self, binary_terms, essential, prime_implicants, coverage):
        minimal = essential.copy()
        remaining = [t for t in binary_terms if not any(self.covers(imp, t) for imp in minimal)]

        for imp in prime_implicants:
            if imp in minimal:
                continue
            for term in remaining:
                if self.covers(imp, term):
                    minimal.append(imp)
                    remaining = [t for t in remaining if not self.covers(imp, t)]
                    break
        return minimal

    def format_output(self, stages, prime_implicants, minimal_implicants):
        output = ["Этапы склеивания:"]
        for stage in stages:
            output.extend(stage['stage_info'])
            terms = [self.binary_term_to_expr(t, False) for t in stage['new_terms']]
            output.append("Результат: " + " & ".join(terms))

        prime_expr = [self.binary_term_to_expr(t, False) for t in prime_implicants]
        minimal_expr = [self.binary_term_to_expr(t, False) for t in minimal_implicants]

        output.append("Простые импликанты: " + " & ".join(prime_expr))
        output.append("Минимальная СКНФ: " + " & ".join(minimal_expr))
        return "\n".join(output)

    def binary_term_to_expr(self, term, is_conjunction=True):
        expr = []
        for var, bit in zip(self.variables, term):
            if bit == '0':
                expr.append(f'!{var}' if is_conjunction else var)
            elif bit == '1':
                expr.append(var if is_conjunction else f'!{var}')
        if is_conjunction:
            return ' & '.join(expr) if expr else '0'
        else:
            return ' | '.join(expr) if expr else '1'

    def minimize_sdnf_quine(self):
        return self.quine_mccluskey_table(is_sdnf=True)

    def minimize_sknf_quine(self):
        return self.quine_mccluskey_table(is_sdnf=False)

    def quine_mccluskey_table(self, is_sdnf=True):
        binary_terms = self.get_binary_terms(is_sdnf)
        if not binary_terms:
            return "0" if is_sdnf else "1"

        prime_implicants, steps = self.find_prime_implicants(binary_terms)

        coverage_table = self.build_coverage_table(prime_implicants, binary_terms)

        result = [
            "Этапы склеивания:",
            *steps,
            "\nТаблица покрытий:",
            self.format_coverage_table(coverage_table, binary_terms),
            "\nМинимальная форма:",
            self.petrick_method(coverage_table, prime_implicants, is_sdnf)
        ]
        return "\n".join(result)

    def find_prime_implicants(self, terms):
        groups = self.initialize_groups(terms)
        steps = []
        prime = set()
        step_num = 1

        while True:
            new_groups, merged, step_info, step_num = self.process_groups(groups, step_num)
            self.collect_unmerged_terms(groups, merged, prime)
            self.update_steps(steps, step_info, groups, merged)

            if not new_groups:
                break

            groups = self.update_groups(new_groups)

        return self.finalize_results(prime, steps)

    def initialize_groups(self, terms):
        groups = {}
        for term in terms:
            cnt = term.count('1')
            groups.setdefault(cnt, []).append((term, (term,)))
        return groups

    def process_groups(self, groups, step_num):
        new_groups = defaultdict(list)
        merged = set()
        step_info = []
        keys = sorted(groups.keys())

        for i in range(len(keys) - 1):
            k1, k2 = keys[i], keys[i + 1]
            if k2 - k1 == 1:
                step_info, step_num = self.process_group_pair(
                    groups[k1],
                    groups[k2],
                    k1,
                    new_groups,
                    merged,
                    step_info,
                    step_num
                )

        return new_groups, merged, step_info, step_num

    def process_group_pair(self, group1, group2, group_key, new_groups, merged, step_info, step_num):
        for t1 in group1:
            for t2 in group2:
                res = self.merge_terms(t1[0], t2[0])
                if res:
                    new_term, pos = res
                    covered_terms = t1[1] + t2[1]
                    new_groups[group_key].append((new_term, covered_terms, pos))
                    merged.update({t1[0], t2[0]})
                    step_info.append(
                        f"Шаг {step_num}: Склеиваем {t1[0]} и {t2[0]} → {new_term} (позиция {pos})"
                    )
                    step_num += 1
        return step_info, step_num

    def collect_unmerged_terms(self, groups, merged, prime):
        for k in sorted(groups):
            for term_info in groups[k]:
                term = term_info[0]
                if term not in merged:
                    prime.add(term_info)

    def update_steps(self, steps, step_info, groups, merged):
        if step_info:
            steps.append("\n".join(step_info))

        unmerged = [term_info[0] for group in groups.values() for term_info in group if term_info[0] not in merged]
        if unmerged:
            steps.append(f"Оставшиеся термы: {', '.join(unmerged)}")

    def update_groups(self, new_groups):
        groups = defaultdict(list)
        for k in new_groups:
            for item in new_groups[k]:
                groups[k].append(item[:2])
        return groups

    def finalize_results(self, prime, steps):
        prime_implicants = {pi[0]: pi[1] for pi in prime}
        steps.append(f"\nИтоговые простые импликанты: {', '.join(prime_implicants)}")
        return prime_implicants, steps

    def merge_terms(self, t1, t2):
        mismatch = 0
        pos = -1
        for i in range(len(t1)):
            if t1[i] != t2[i]:
                mismatch += 1
                pos = i
                if mismatch > 1:
                    return None
        if mismatch == 1:
            return t1[:pos] + 'X' + t1[pos + 1:], pos
        return None

    def build_coverage_table(self, prime_implicants, terms):
        table = {}
        for imp, covered in prime_implicants.items():
            table[imp] = [1 if t in covered else 0 for t in terms]
        return table

    def format_coverage_table(self, table, terms):
        header = "Импликант/Терм | " + " | ".join(terms)
        separator = "-" * len(header)
        rows = []
        for imp, coverage in table.items():
            row = f"{imp:15} | " + " | ".join(f"{c:^{len(t)}}"
                                              for c, t in zip(coverage, terms))
            rows.append(row)
        return "\n".join([header, separator] + rows)

    def petrick_method(self, table, prime_implicants, is_sdnf):

        essential = set()
        remaining = set(prime_implicants.keys())


        for col in range(len(table[next(iter(table))])):
            covering = [imp for imp, cov in table.items() if cov[col]]
            if len(covering) == 1:
                essential.add(covering[0])
                remaining.discard(covering[0])

        return self.format_implicants(
            list(essential) + list(remaining),
            is_sdnf
        )

    def format_implicants(self, implicants, is_sdnf):
        operator = ' | ' if is_sdnf else ' & '
        terms = []
        if is_sdnf:
            for imp in implicants:
                term = []
                for i, c in enumerate(imp):
                    var = self.variables[i]
                    if c == '1':
                        term.append(var)
                    elif c == '0':
                        term.append(f"!{var}")
                if term:
                    terms.append(f"({' & '.join(term)})")
        else:
            for imp in implicants:
                term = []
                for i, c in enumerate(imp):
                    var = self.variables[i]
                    if c == '0':
                        term.append(var)
                    elif c == '1':
                        term.append(f"!{var}")
                if term:
                    terms.append(f"({' | '.join(term)})")
        return operator.join(terms)

    def build_karnaugh_map(self):
        num_vars = self._validate_kmap_variables()
        rows, cols = self._get_kmap_dimensions(num_vars)

        row_gray = self._generate_gray_code_sequence(rows)
        col_gray = self._generate_gray_code_sequence(cols) if cols > 1 else ['0']

        kmap = self._initialize_kmap(rows, cols)
        self._fill_kmap_values(kmap, row_gray, col_gray, num_vars)

        return kmap, row_gray, col_gray

    def _validate_kmap_variables(self) -> int:
        num_vars = len(self.variables)
        if num_vars > 6:
            raise ValueError("Karnaugh maps are only supported for up to 6 variables")
        return num_vars

    def _get_kmap_dimensions(self, num_vars: int) -> tuple:
        dimensions = {
            1: (2, 1),
            2: (2, 2),
            3: (2, 4),
            4: (4, 4),
            5: (4, 8),
            6: (8, 8)
        }
        return dimensions[num_vars]

    def _generate_gray_code_sequence(self, length: int) -> list:
        if length == 0:
            return []
        bits = int(math.log2(length))
        return [bin(num ^ (num >> 1))[2:].zfill(bits) for num in range(length)]

    def _initialize_kmap(self, rows: int, cols: int) -> list:
        return [[0 for _ in range(cols)] for _ in range(rows)]

    def _fill_kmap_values(self, kmap: list, row_gray: list, col_gray: list, num_vars: int):
        for values, result in self.truth_table:
            binary_str = ''.join(map(str, values))

            row_part = binary_str[:int(math.log2(len(row_gray)))]
            col_part = binary_str[int(math.log2(len(row_gray))):] if num_vars > 1 else '0'

            try:
                row_idx = row_gray.index(row_part)
                col_idx = col_gray.index(col_part) if num_vars > 1 else 0
                kmap[row_idx][col_idx] = int(result)
            except ValueError:
                continue

    def minimize_sdnf_karnaugh(self):
        kmap, row_gray, col_gray = self.build_karnaugh_map()

        kmap_visualization = self.print_karnaugh_map(kmap, row_gray, col_gray, highlight_value=1)

        input_terms = [i for i, (_, result) in enumerate(self.truth_table) if result]
        optimized = self.optimize_kmap(input_terms, len(self.variables), self.variables, True)
        optimized_expr = " | ".join(optimized) if optimized else "0"

        result = [
            "Карта Карно для СДНФ (выделены 1):",
            kmap_visualization,
            "\nМинимальная СДНФ:",
            optimized_expr
        ]
        return "\n".join(result)

    def minimize_sknf_karnaugh(self):
        kmap, row_gray, col_gray = self.build_karnaugh_map()

        kmap_visualization = self.print_karnaugh_map(kmap, row_gray, col_gray, highlight_value=0)

        input_terms = [i for i, (_, result) in enumerate(self.truth_table) if not result]
        optimized = self.optimize_kmap(input_terms, len(self.variables), self.variables, False)
        optimized_expr = " & ".join(optimized) if optimized else "1"

        result = [
            "Карта Карно для СКНФ (выделены 0):",
            kmap_visualization,
            "\nМинимальная СКНФ:",
            optimized_expr
        ]
        return "\n".join(result)

    def print_karnaugh_map(self, kmap, row_gray, col_gray, highlight_value=None):
        header = " " * len(row_gray[0]) + " | " + " | ".join(col_gray)
        separator = "-" * len(header)

        rows = []
        for i, row in enumerate(kmap):
            row_label = row_gray[i]
            row_str = f"{row_label} | " + " | ".join(
                f"\033[1;31m{val}\033[0m" if highlight_value is not None and val == highlight_value else str(val)
                for val in row)
            rows.append(row_str)

        return "\n".join([header, separator] + rows)

    def generate_gray_sequence(self, length: int) -> list:
        if not isinstance(length, int) or length <= 0:
            raise ValueError("Length must be a positive integer")
        return [num ^ (num >> 1) for num in range(1 << length)]

    def _convert_to_binary_terms(self, input_terms: list, variable_count: int) -> list:
        if not isinstance(input_terms, list):
            raise TypeError("Input terms must be a list")
        if not isinstance(variable_count, int) or variable_count <= 0:
            raise ValueError("Variable count must be a positive integer")
        if any(not isinstance(term, int) or term < 0 for term in input_terms):
            raise ValueError("All terms must be non-negative integers")

        max_term = max(input_terms, default=0)
        if max_term >= (1 << variable_count):
            raise ValueError(f"Term {max_term} exceeds maximum value for {variable_count} variables")

        return [tuple((term >> i) & 1 for i in reversed(range(variable_count))) for term in input_terms]

    def _combine_terms_pair(self, term_a: tuple, term_b: tuple) -> tuple:
        mismatch_count = 0
        combined = []
        for bit_a, bit_b in zip(term_a, term_b):
            combined.append(bit_a if bit_a == bit_b else '-')
            mismatch_count += (bit_a != bit_b)
        return tuple(combined) if mismatch_count == 1 else None

    def _identify_prime_implicants(self, binary_terms: list, var_count: int) -> set:
        self._validate_binary_terms_input(binary_terms, var_count)

        grouped_terms = self._group_terms_by_bit_count(binary_terms, var_count)
        prime_set = set()
        unchecked_terms = set(binary_terms)

        while grouped_terms:
            next_group, processed = self._process_term_groups(grouped_terms)
            prime_set.update(unchecked_terms - processed)
            unchecked_terms = self._flatten_next_group(next_group)
            grouped_terms = self._filter_non_empty_groups(next_group)

        return prime_set

    def _validate_binary_terms_input(self, binary_terms: list, var_count: int) -> None:
        if not isinstance(binary_terms, list):
            raise TypeError("Binary terms must be a list")
        if not binary_terms:
            raise ValueError("Binary terms list cannot be empty")
        if not isinstance(var_count, int) or var_count <= 0:
            raise ValueError("Variable count must be a positive integer")

    def _group_terms_by_bit_count(self, binary_terms: list, var_count: int) -> dict:
        grouped_terms = {}
        for term in binary_terms:
            if not isinstance(term, tuple) or len(term) != var_count:
                raise ValueError(f"Term {term} has invalid format or length")
            key = term.count(1)
            grouped_terms.setdefault(key, []).append(term)
        return grouped_terms

    def _process_term_groups(self, grouped_terms: dict) -> tuple:
        next_group = {}
        processed = set()

        for key in sorted(grouped_terms):
            for t1 in grouped_terms[key]:
                for t2 in grouped_terms.get(key + 1, []):
                    merged = self._combine_terms_pair(t1, t2)
                    if merged:
                        processed.update({t1, t2})
                        next_group.setdefault(merged.count(1), []).append(merged)

        return next_group, processed

    def _flatten_next_group(self, next_group: dict) -> set:
        return set(sum(next_group.values(), []))

    def _filter_non_empty_groups(self, grouped_terms: dict) -> dict:
        return {k: v for k, v in grouped_terms.items() if v}

    def _is_implicant_covering(self, implicant: tuple, minterm: tuple) -> bool:
        if not isinstance(implicant, tuple) or not isinstance(minterm, tuple):
            raise TypeError("Both arguments must be tuples")
        if len(implicant) != len(minterm):
            raise ValueError("Implicant and minterm not equal")

        return all(imp_bit == '-' or imp_bit == mt_bit for imp_bit, mt_bit in zip(implicant, minterm))

    def _select_essential_implicants(self, prime_set: set, minterms: list) -> set:
        self._validate_prime_implicants_input(prime_set, minterms)

        coverage_table = self._build_coverage_table(prime_set, minterms)
        essential = self._find_initially_essential(prime_set, minterms, coverage_table)
        covered = self._get_covered_minterms(essential, coverage_table)

        remaining = set(minterms) - covered
        essential.update(self._select_remaining_implicants(coverage_table, remaining))

        return essential

    def _validate_prime_implicants_input(self, prime_set: set, minterms: list) -> None:
        if not isinstance(prime_set, set):
            raise TypeError("Prime set must be a set")
        if not isinstance(minterms, list):
            raise TypeError("Minterms must be a list")
        if not minterms:
            raise ValueError("Minterms list cannot be empty")

    def _build_coverage_table(self, prime_set: set, minterms: list) -> dict:
        return {
            imp: [mt for mt in minterms if self._is_implicant_covering(imp, mt)]
            for imp in prime_set
        }

    def _find_initially_essential(self, prime_set: set, minterms: list, coverage_table: dict) -> set:
        essential = set()

        for mt in minterms:
            covering = [imp for imp in prime_set if self._is_implicant_covering(imp, mt)]
            if not covering:
                raise ValueError(f"Minterm {mt} is not covered by any prime implicant")
            if len(covering) == 1:
                essential.add(covering[0])

        return essential

    def _get_covered_minterms(self, essential_implicants: set, coverage_table: dict) -> set:
        covered = set()
        for imp in essential_implicants:
            covered.update(coverage_table[imp])
        return covered

    def _select_remaining_implicants(self, coverage_table: dict, remaining_minterms: set) -> set:
        selected = set()

        while remaining_minterms:
            best_imp, best_covered = max(
                coverage_table.items(),
                key=lambda item: len(set(item[1]) & remaining_minterms)
            )

            selected.add(best_imp)
            remaining_minterms -= set(best_covered)

        return selected

    def optimize_kmap(self, input_terms: list, var_count: int, var_names: list, use_conjunctive: bool = True) -> list:
        self._validate_inputs(input_terms, var_count, var_names)

        binary_terms = self._convert_to_binary_terms(input_terms, var_count)
        primes = self._identify_prime_implicants(binary_terms, var_count)
        essentials = self._select_essential_implicants(primes, binary_terms)

        return self._build_output_expression(list(essentials), var_names, use_conjunctive)

    def _validate_inputs(self, input_terms: list, var_count: int, var_names: list) -> None:
        if not isinstance(input_terms, list):
            raise TypeError("Input terms must be a list")
        if not isinstance(var_count, int) or var_count <= 0:
            raise ValueError("Variable count must be a positive integer")
        if not isinstance(var_names, list) or len(var_names) != var_count:
            raise ValueError("Variable names must be a list matching variable count")
        if any(not isinstance(name, str) for name in var_names):
            raise ValueError("All variable names must be strings")

    def _build_output_expression(self, implicants: list, var_names: list, use_conjunctive: bool) -> list:
        operator = '&' if use_conjunctive else '|'
        result = []

        for imp in implicants:
            components = self._build_implicant_components(imp, var_names, use_conjunctive)
            if components:  # Only add if not empty
                result.append(f"({operator.join(components)})")

        return result

    def _build_implicant_components(self, implicant: str, var_names: list, use_conjunctive: bool) -> list:
        components = []

        for var, bit in zip(var_names, implicant):
            if bit == '-':
                continue

            if use_conjunctive:
                component = var if bit else f"!{var}"
            else:
                component = f"!{var}" if bit else var

            components.append(component)

        return components

    def validate_variable_count(self, num_vars):
        if num_vars > 6:
            raise ValueError("Karnaugh maps are only supported for up to 6 variables")



