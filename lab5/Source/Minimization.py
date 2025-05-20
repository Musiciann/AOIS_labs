from Logical_Operations import LogicalOperations

class KarnoMap(LogicalOperations):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        self.var_count = len(labels)
        self.order = [0, 1, 3, 2]

    def get_biggest_groups(self, groups):
        biggest = []
        for i, (cells_i, pat_i) in enumerate(groups):
            is_biggest = True
            for j, (cells_j, pat_j) in enumerate(groups):
                if i != j and cells_i <= cells_j and cells_i != cells_j:
                    is_biggest = False
                    break
            if is_biggest:
                biggest.append((cells_i, pat_i))
        return biggest

    def choose_groups(self, big_groups, ones_pos):
        chosen = []
        to_cover = set(range(len(ones_pos)))
        for i, pos in enumerate(ones_pos):
            covers = [idx for idx, (cells, _) in enumerate(big_groups) if pos in cells]
            if len(covers) == 1:
                idx = covers[0]
                if idx not in chosen:
                    chosen.append(idx)
                    for j, other_pos in enumerate(ones_pos):
                        if other_pos in big_groups[idx][0]:
                            to_cover.discard(j)
        while to_cover:
            best_idx = None
            best_covered = set()
            for idx, (cells, _) in enumerate(big_groups):
                if idx in chosen:
                    continue
                covered = {j for j in to_cover if ones_pos[j] in cells}
                if len(covered) > len(best_covered):
                    best_covered = covered
                    best_idx = idx
            if best_idx is None:
                print("Не удалось покрыть все единицы!")
                break
            chosen.append(best_idx)
            to_cover -= best_covered
        return chosen

    def make_result(self, chosen, big_groups):
        terms = set()
        for idx in chosen:
            pattern = big_groups[idx][1]
            term = []
            for i, val in enumerate(pattern):
                if val == '1':
                    term.append(self.labels[i])
                elif val == '0':
                    term.append(f'!{self.labels[i]}')
            if term:
                term_str = term[0] if len(term) == 1 else '(' + ' & '.join(sorted(term)) + ')'
                terms.add(term_str)
        terms = sorted(terms)
        return "0" if not terms else " | ".join(terms)

    def make_map_1_to_4(self):
        ones = []
        if self.var_count == 1:
            grid = [0] * 2
            for row in self.inputs:
                a, = row[0]
                grid[a] = row[1]
                if row[1] == 1:
                    ones.append((a,))
        elif self.var_count == 2:
            grid = [[0] * 2 for _ in range(2)]
            for row in self.inputs:
                a, b = row[0]
                grid[a][b] = row[1]
                if row[1] == 1:
                    ones.append((a, b))
        elif self.var_count == 3:
            grid = [[0] * 4 for _ in range(2)]
            for row in self.inputs:
                a, b, c = row[0]
                grid[a][self.order.index((b << 1) | c)] = row[1]
                if row[1] == 1:
                    ones.append((a, b, c))
        else:
            grid = [[0] * 4 for _ in range(4)]
            for row in self.inputs:
                a, b, c, d = row[0]
                grid[self.order.index((a << 1) | b)][self.order.index((c << 1) | d)] = row[1]
                if row[1] == 1:
                    ones.append((a, b, c, d))
        return grid, ones

    def show_map_1_to_4(self, grid):
        print("Минимизация (карты Карно):")
        if self.var_count == 1:
            print("a    0  1")
            print("     " + "  ".join(str(grid[i]) for i in range(2)))
        elif self.var_count == 2:
            print("a\\b   0  1")
            for r in range(2):
                print(f"{r}     " + "  ".join(str(grid[r][c]) for c in range(2)))
        elif self.var_count == 3:
            print("a\\bc   00  01  11  10")
            for r in range(2):
                print(f"{r}     " + "  ".join(f"{grid[r][c]:>3}" for c in range(4)))
        else:
            print("ab\\cd   00  01  11  10")
            for r in range(4):
                label = format(self.order[r], '02b')
                print(f"{label}     " + "  ".join(f"{grid[r][c]:>3}" for c in range(4)))

    def get_cells_1_to_4(self, grid, r_start, r_size, c_start, c_size, r_max, c_max):
        cells = set()
        all_ones = True
        for dr in range(r_size):
            r = (r_start + dr) % r_max
            for dc in range(c_size):
                c = (c_start + dc) % c_max
                cells.add((r, c))
                val = grid[c] if self.var_count == 1 else grid[r][c]
                if val == 0:
                    all_ones = False
                    break
            if not all_ones:
                break
        return cells, all_ones

    def make_pattern_1_to_4(self, cells):
        pattern = ['-'] * self.var_count
        values = []
        for r, c in cells:
            if self.var_count == 1:
                values.append([c])
            elif self.var_count == 2:
                values.append([r, c])
            elif self.var_count == 3:
                bc = self.order[c]
                values.append([r, (bc >> 1) & 1, bc & 1])
            else:
                ab = self.order[r]
                cd = self.order[c]
                values.append([(ab >> 1) & 1, ab & 1, (cd >> 1) & 1, cd & 1])
        for i in range(self.var_count):
            unique_vals = {val[i] for val in values}
            if len(unique_vals) == 1:
                pattern[i] = str(unique_vals.pop())
        return pattern

    def find_groups_1_to_4(self, grid, r_max, c_max):
        groups = []
        row_sizes = [1, 2] if r_max <= 2 else [1, 2, 4]
        col_sizes = [1, 2] if c_max == 2 else [1, 2, 4]
        for r_size in row_sizes:
            for c_size in col_sizes:
                for r_start in range(r_max):
                    for c_start in range(c_max):
                        cells, all_ones = self.get_cells_1_to_4(grid, r_start, r_size, c_start, c_size, r_max, c_max)
                        if all_ones and cells:
                            pattern = self.make_pattern_1_to_4(cells)
                            groups.append((cells, pattern))
        return groups

    def get_ones_pos_1_to_4(self, ones):
        positions = []
        for p in ones:
            if self.var_count == 1:
                positions.append((0, p[0]))
            elif self.var_count == 2:
                positions.append((p[0], p[1]))
            elif self.var_count == 3:
                positions.append((p[0], self.order.index((p[1] << 1) | p[2])))
            else:
                positions.append((self.order.index((p[0] << 1) | p[1]), self.order.index((p[2] << 1) | p[3])))
        return positions

    def karno_1_to_4_sdnf(self):
        if self.var_count not in [1, 2, 3, 4]:
            return "Метод работает только с 1, 2, 3 или 4 переменными."
        grid, ones = self.make_map_1_to_4()
        self.show_map_1_to_4(grid)
        if not ones:
            return "0"
        r_max = 1 if self.var_count == 1 else 2 if self.var_count in [2, 3] else 4
        c_max = 2 if self.var_count in [1, 2] else 4
        groups = self.find_groups_1_to_4(grid, r_max, c_max)
        big_groups = self.get_biggest_groups(groups)
        ones_pos = self.get_ones_pos_1_to_4(ones)
        chosen = self.choose_groups(big_groups, ones_pos)
        print('\n')
        return self.make_result(chosen, big_groups)