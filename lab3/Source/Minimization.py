from collections import defaultdict
from LF import LogicalFunction
import math

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
        num_vars = len(self.variables)
        self.validate_variable_count(num_vars)

        rows, cols = self.determine_map_dimensions(num_vars)
        kmap = self.create_empty_map(rows, cols)
        row_gray, col_gray = self.generate_gray_codes(rows, cols)

        self.fill_kmap_with_truth_table(kmap, row_gray, col_gray, num_vars)
        self.fill_remaining_cells(kmap)

        return kmap, row_gray, col_gray

    def validate_variable_count(self, num_vars):
        if num_vars > 6:
            raise ValueError("Karnaugh maps are only supported for up to 6 variables")

    def determine_map_dimensions(self, num_vars):
        dimensions = {
            1: (2, 1),
            2: (2, 2),
            3: (2, 4),
            4: (4, 4),
            5: (4, 8),
            6: (8, 8)
        }
        return dimensions[num_vars]

    def create_empty_map(self, rows, cols):
        return [[None for _ in range(cols)] for _ in range(rows)]

    def generate_gray_codes(self, rows, cols):
        row_gray = self.gray_code(int(math.log2(rows)))
        col_gray = self.gray_code(int(math.log2(cols))) if cols > 1 else ['0']
        return row_gray, col_gray

    def fill_kmap_with_truth_table(self, kmap, row_gray, col_gray, num_vars):
        for (values, result) in self.truth_table:
            row_vars, col_vars = self.split_variables(values, num_vars)
            self.set_kmap_value(kmap, row_gray, col_gray, row_vars, col_vars, result)

    def split_variables(self, values, num_vars):
        split_rules = {
            1: (slice(0, 1), slice(0, 0)),
            2: (slice(0, 1), slice(1, 2)),
            3: (slice(0, 1), slice(1, 3)),
            4: (slice(0, 2), slice(2, 4)),
            5: (slice(0, 2), slice(2, 5)),
            6: (slice(0, 3), slice(3, 6))
        }
        row_slice, col_slice = split_rules[num_vars]
        return values[row_slice], values[col_slice]

    def set_kmap_value(self, kmap, row_gray, col_gray, row_vars, col_vars, result):
        row_bits = ''.join(map(str, row_vars))
        col_bits = ''.join(map(str, col_vars)) if col_vars else '0'

        try:
            row_idx = row_gray.index(row_bits)
            col_idx = col_gray.index(col_bits) if col_bits else 0
            kmap[row_idx][col_idx] = int(result)
        except ValueError:
            pass

    def fill_remaining_cells(self, kmap):
        for i in range(len(kmap)):
            for j in range(len(kmap[0])):
                if kmap[i][j] is None:
                    kmap[i][j] = 0

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

    def minimize_sdnf_karnaugh(self):
        kmap, row_gray, col_gray = self.build_karnaugh_map()
        karnaugh_map_str = self.print_karnaugh_map(kmap, row_gray, col_gray, highlight_value=1)

        rectangles = self.find_optimal_rectangles(kmap, target=1)
        terms = [self.rectangle_to_expression(rect, row_gray, col_gray, is_sdnf=True)
                 for rect in rectangles]

        result = [
            "Карта Карно (выделены 1 для СДНФ):",
            karnaugh_map_str,
            "\nВыбранные прямоугольники:",
            "\n".join([f"Прямоугольник {i + 1}: строка {rect[0]}, колонка {rect[1]}, высота {rect[2]}, ширина {rect[3]}"
                       for i, rect in enumerate(rectangles)]),
            "\nМинимальная СДНФ:",
            " ∨ ".join([f"({term})" for term in terms]) if terms else "0"
        ]
        return "\n".join(result)

    def minimize_sknf_karnaugh(self):
        kmap, row_gray, col_gray = self.build_karnaugh_map()
        karnaugh_map_str = self.print_karnaugh_map(kmap, row_gray, col_gray, highlight_value=0)

        rectangles = self.find_optimal_rectangles(kmap, target=0)
        terms = [self.rectangle_to_expression(rect, row_gray, col_gray, is_sdnf=False)
                 for rect in rectangles]

        result = [
            "Карта Карно (выделены 0 для СКНФ):",
            karnaugh_map_str,
            "\nВыбранные прямоугольники:",
            "\n".join([f"Прямоугольник {i + 1}: строка {rect[0]}, колонка {rect[1]}, высота {rect[2]}, ширина {rect[3]}"
                       for i, rect in enumerate(rectangles)]),
            "\nМинимальная СКНФ:",
            " & ".join([f"({term})" for term in terms]) if terms else "1"
        ]
        return "\n".join(result)

    def gray_code(self, n):
        if n == 0:
            return ['']
        first_half = self.gray_code(n - 1)
        second_half = first_half.copy()
        second_half.reverse()

        return ['0' + code for code in first_half] + ['1' + code for code in second_half]

    def find_optimal_rectangles(self, kmap, target=1):
        rows = len(kmap)
        cols = len(kmap[0]) if rows > 0 else 0
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        rectangles = []

        all_rectangles = []
        for i in range(rows):
            for j in range(cols):
                if kmap[i][j] == target:
                    rects = self.find_all_rectangles_at(kmap, i, j, target)
                    all_rectangles.extend(rects)

        unique_rectangles = []
        seen = set()
        for rect in all_rectangles:
            rect_key = (rect[0], rect[1], rect[2], rect[3])
            if rect_key not in seen:
                seen.add(rect_key)
                unique_rectangles.append(rect)

        unique_rectangles.sort(key=lambda r: r[2]*r[3], reverse=True)

        return self.select_minimal_cover(unique_rectangles, kmap, target)

    def find_all_rectangles_at(self, kmap, i, j, target):

        rows = len(kmap)
        cols = len(kmap[0])
        rectangles = []

        max_height = min(2**int(math.log2(rows)), rows)
        max_width = min(2**int(math.log2(cols)), cols)

        for height in [2**p for p in range(int(math.log2(max_height)) + 1)]:
            for width in [2**p for p in range(int(math.log2(max_width)) + 1)]:
                valid = True
                for h in range(height):
                    for w in range(width):
                        row = (i + h) % rows
                        col = (j + w) % cols
                        if kmap[row][col] != target:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    rectangles.append((i, j, height, width))

        return rectangles

    def select_minimal_cover(self, rectangles, kmap, target):
        target_cells = self.get_target_cells(kmap, target)
        if not target_cells:
            return []

        coverage = self.build_cell_coverage(rectangles, kmap, target_cells)
        essential = self.find_essential_rectangles(coverage)
        selected, covered = self.apply_essential_rectangles(essential, kmap, target_cells)

        remaining = [r for r in rectangles if r not in selected]
        remaining.sort(key=lambda r: r[2] * r[3], reverse=True)

        selected = self.select_remaining_rectangles(remaining, selected, covered, kmap, target_cells)
        return selected

    def get_target_cells(self, kmap, target):
        rows = len(kmap)
        cols = len(kmap[0]) if rows > 0 else 0
        return {(i, j) for i in range(rows) for j in range(cols)
                if kmap[i][j] == target}

    def build_cell_coverage(self, rectangles, kmap, target_cells):
        coverage = defaultdict(list)
        for rect in rectangles:
            covered_cells = self.get_covered_cells(rect, kmap, target_cells)
            for cell in covered_cells:
                coverage[cell].append(rect)
        return coverage

    def get_covered_cells(self, rect, kmap, target_cells):
        i, j, height, width = rect
        rows = len(kmap)
        cols = len(kmap[0]) if rows > 0 else 0
        covered = set()

        for h in range(height):
            for w in range(width):
                row = (i + h) % rows
                col = (j + w) % cols
                if (row, col) in target_cells:
                    covered.add((row, col))
        return covered

    def find_essential_rectangles(self, coverage):
        essential = set()
        for cell, rects in coverage.items():
            if len(rects) == 1:
                essential.add(rects[0])
        return essential

    def apply_essential_rectangles(self, essential, kmap, target_cells):
        selected = list(essential)
        covered = set()

        for rect in selected:
            covered.update(self.get_covered_cells(rect, kmap, target_cells))

        return selected, covered

    def select_remaining_rectangles(self, remaining, selected, covered, kmap, target_cells):
        while len(covered) < len(target_cells):
            best_rect, best_new = None, 0

            for rect in remaining:
                new_cover = self.calculate_new_coverage(rect, covered, kmap, target_cells)
                if new_cover > best_new:
                    best_new = new_cover
                    best_rect = rect

            if not best_rect:
                break

            selected.append(best_rect)
            remaining.remove(best_rect)
            covered.update(self.get_covered_cells(best_rect, kmap, target_cells))

        return selected

    def calculate_new_coverage(self, rect, covered, kmap, target_cells):
        new_cover = 0
        for cell in self.get_covered_cells(rect, kmap, target_cells):
            if cell not in covered:
                new_cover += 1
        return new_cover

    def find_largest_power2_rectangle(self, kmap, start_row, start_col, target):
        rows = len(kmap)
        cols = len(kmap[0])
        max_rect = None
        max_area = 0
        for height_pow in range(int(math.log2(rows)) + 1):
            height = 2 ** height_pow
            if height > rows:
                continue
            for width_pow in range(int(math.log2(cols)) + 1):
                width = 2 ** width_pow
                if width > cols:
                    continue

                valid = True
                for h in range(height):
                    for w in range(width):
                        row = (start_row + h) % rows
                        col = (start_col + w) % cols
                        if kmap[row][col] != target:
                            valid = False
                            break
                    if not valid:
                        break
                if valid and height * width > max_area:
                    max_area = height * width
                    max_rect = (start_row, start_col, height, width)

        return max_rect

    def mark_rectangle_visited(self, visited, rectangle):
        i, j, height, width = rectangle
        rows = len(visited)
        cols = len(visited[0])

        for h in range(height):
            for w in range(width):
                visited[(i + h) % rows][(j + w) % cols] = True


    def rectangle_to_expression(self, rectangle, row_gray, col_gray, is_sdnf=True):
        i, j, height, width = rectangle
        rows = len(row_gray)
        cols = len(col_gray) if row_gray else 0

        row_expr = self.process_row_variables(i, height, rows, row_gray, is_sdnf)
        col_expr = self.process_col_variables(j, width, cols, col_gray, row_gray, is_sdnf)

        var_expr = row_expr + col_expr

        return self.format_expression(var_expr, is_sdnf)


    def process_row_variables(self, start_row, height, total_rows, row_gray, is_sdnf):
        if total_rows <= 1:
            return []

        common_bits = self.find_common_bits(
            start_row, height, total_rows,
            row_gray, len(row_gray[0])
        )

        return [
            self.bit_to_variable(bit_idx, bit, is_sdnf, bit_idx)
            for bit_idx, bit in common_bits
        ]


    def process_col_variables(self, start_col, width, total_cols, col_gray, row_gray, is_sdnf):
        if total_cols <= 1:
            return []

        common_bits = self.find_common_bits(
            start_col, width, total_cols,
            col_gray, len(col_gray[0])
        )

        return [
            self.bit_to_variable(
                bit_idx, bit, is_sdnf,
                len(row_gray[0]) + bit_idx if row_gray else bit_idx
            )
            for bit_idx, bit in common_bits
        ]

    def find_common_bits(self, start, size, total, gray_code, bits_count):
        common_bits = []
        for bit_idx in range(bits_count):
            first_bit = gray_code[start][bit_idx]
            same_bit = all(
                gray_code[(start + offset) % total][bit_idx] == first_bit
                for offset in range(size)
            )
            if same_bit:
                common_bits.append((bit_idx, first_bit))
        return common_bits


    def bit_to_variable(self, bit_idx, bit, is_sdnf, var_idx):
        var = self.variables[var_idx]
        if is_sdnf:
            return var if bit == '1' else f"!{var}"
        else:
            return var if bit == '0' else f"!{var}"


    def format_expression(self, var_expr, is_sdnf):
        if not var_expr:
            return "1" if is_sdnf else "0"

        if is_sdnf:
            return " & ".join(var_expr)
        else:
            return " | ".join(var_expr)

