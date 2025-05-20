from Minimize_Calculational import minimize_sdnf_by_calculation_method

def bcd_to_bcd_plus_6(a, b, c, s):
    decimal_input = (a << 3) | (b << 2) | (c << 1) | s
    decimal_output = decimal_input + 6
    a_out = (decimal_output >> 3) & 1
    b_out = (decimal_output >> 2) & 1
    c_out = (decimal_output >> 1) & 1
    s_out = decimal_output & 1
    return a_out, b_out, c_out, s_out

def binary_adder_3_input(a, b, c):
    sum_bit = (a + b + c) % 2
    carry_bit = (a + b + c) // 2
    return sum_bit, carry_bit

def generate_bcd_conversion_table():
    table = []
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for s in range(2):
                    decimal_input = (a << 3) | (b << 2) | (c << 1) | s
                    if decimal_input > 9:
                        continue
                    a_out, b_out, c_out, s_out = bcd_to_bcd_plus_6(a, b, c, s)
                    table.append([a, b, c, s, a_out, b_out, c_out, s_out])
    return table

def generate_adder_truth_table():
    table = []
    for a in range(2):
        for b in range(2):
            for c in range(2):
                sum_bit, carry_bit = binary_adder_3_input(a, b, c)
                table.append([a, b, c, sum_bit, carry_bit])
    return table

def print_main_bcd_table(table):
    print(f"{'Вход (Д8421)':<15}{'Выход (Д8421+6)':<15}")
    print("-" * 30)
    for row in table:
        input_bits = ''.join(str(bit) for bit in row[:4])
        output_bits = ''.join(str(bit) for bit in row[4:])
        print(f"{input_bits:<15}{output_bits:<15}")

def print_truth_table(table, input_names, output_names, title):
    print(f"\n{title}")
    header = " ".join(input_names) + " | " + " ".join(output_names)
    print(header)
    print("-" * (len(header) + 2))
    for row in table:
        inputs = ' '.join(str(bit) for bit in row[:len(input_names)])
        outputs = ' '.join(str(bit) for bit in row[len(input_names):])
        print(f"{inputs} | {outputs}")

def build_sdnf(table, input_count, output_index, var_names):
    terms = []
    for row in table:
        inputs = row[:input_count]
        output_bit = row[input_count + output_index]
        if output_bit == 1:
            term = []
            for i, val in enumerate(inputs):
                term.append(f"{var_names[i]}" if val else f"!{var_names[i]}")
            terms.append(f"({'&'.join(term)})")
    return '|'.join(terms)

if __name__ == '__main__':
    print("=" * 50)
    print("Преобразователь BCD в BCD+6")
    print("=" * 50)
    bcd_table = generate_bcd_conversion_table()
    print_main_bcd_table(bcd_table)

    for i in range(4):
        print_truth_table(bcd_table, ['A', 'B', 'C', 'D'], ['A', 'B', 'C', 'D'],
                          f"Таблица истинности для выхода {['A', 'B', 'C', 'D'][i]}:")
        sdnf = build_sdnf(bcd_table, 4, i, ['A', 'B', 'C', 'D'])
        print(f"\nСДНФ для выхода {['A', 'B', 'C', 'D'][i]}:")
        print(sdnf)
        minimized = minimize_sdnf_by_calculation_method(sdnf)
        print(f"Минимизированное выражение для {['A', 'B', 'C', 'D'][i]}:")
        print(minimized)

    print("\n" + "=" * 50)
    print("Одноразрядный двоичный сумматор на 3 входа")
    print("=" * 50)
    adder_table = generate_adder_truth_table()

    print_truth_table(adder_table, ['A', 'B', 'C'], ['S', 'Carry'],
                      "Таблица истинности для сумматора (выход S):")
    sdnf_sum = build_sdnf(adder_table, 3, 0, ['A', 'B', 'C'])
    print(f"\nСДНФ для выхода S:")
    print(sdnf_sum)
    minimized_sum = minimize_sdnf_by_calculation_method(sdnf_sum)
    print(f"Минимизированное выражение для S:")
    print(minimized_sum)

    print_truth_table(adder_table, ['A', 'B', 'C'], ['S', 'Carry'],
                      "Таблица истинности для сумматора (выход Carry):")
    sdnf_carry = build_sdnf(adder_table, 3, 1, ['A', 'B', 'C'])
    print(f"\nСДНФ для выхода Carry:")
    print(sdnf_carry)
    minimized_carry = minimize_sdnf_by_calculation_method(sdnf_carry)
    print(f"Минимизированное выражение для Carry:")
    print(minimized_carry)