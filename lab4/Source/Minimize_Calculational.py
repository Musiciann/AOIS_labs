from itertools import product, combinations
from collections import defaultdict

def truth_table_pdnf(expression, variables):
    translation_table = str.maketrans({
        '!': 'not ',
        '&': ' and ',
        '|': ' or ',
        '~': 'not '
    })
    normalized_expr = expression.translate(translation_table)

    table = []
    for values in product([0, 1], repeat=len(variables)):
        env = dict(zip(variables, map(bool, values)))
        try:
            result = eval(normalized_expr, {}, env)
            if result:
                table.append(values)
        except:
            continue
    return table

def to_sdnf(table, variables):
    terms = []
    for row in table:
        term_parts = []
        for var, val in zip(variables, row):
            term_parts.append(f"!{var}" if not val else var)
        terms.append(f"({' & '.join(term_parts)})")
    return ' | '.join(terms)


def minimize_sdnf_by_calculation_method(expression_str):
    variables = sorted({c for c in expression_str if c.isalpha()})
    table = truth_table_pdnf(expression_str, variables)

    if not table:
        return "0"

    terms = [''.join(str(bit) for bit in row) for row in table]

    groups = defaultdict(list)
    for term in terms:
        groups[term.count('1')].append(term)

    prime_implicants = set()
    while True:
        new_groups = defaultdict(list)
        used = set()
        new_primes = set()

        for count in sorted(groups.keys()):
            for term1 in groups[count]:
                for term2 in groups.get(count + 1, []):
                    diff_pos = [i for i in range(len(term1))
                                if term1[i] != term2[i]]
                    if len(diff_pos) == 1:
                        pos = diff_pos[0]
                        new_term = term1[:pos] + '-' + term1[pos + 1:]
                        new_groups[new_term.count('1')].append(new_term)
                        used.add(term1)
                        used.add(term2)

        for group in groups.values():
            for term in group:
                if term not in used:
                    new_primes.add(term)

        prime_implicants.update(new_primes)

        if not new_groups:
            break

        groups = new_groups

    minimized_terms = []
    for prime in prime_implicants:
        term_parts = []
        for var, bit in zip(variables, prime):
            if bit == '0':
                term_parts.append(f"!{var}")
            elif bit == '1':
                term_parts.append(var)
        minimized_terms.append(f"({' & '.join(term_parts)})")

    return ' | '.join(minimized_terms) if minimized_terms else "0"