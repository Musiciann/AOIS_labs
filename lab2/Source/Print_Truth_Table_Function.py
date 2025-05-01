def print_truth_table(truth_table, variables, expression) -> None:
    print("Truth Table")
    print("\t".join(variables) + "\t" + f"{expression}")
    print("----------------------------------------")

    for var, result in truth_table:
        print("\t".join(map(str, var)) + "\t" + str(int(result)))

def show_truth_table_result(binary_result: str, decimal_result: int) -> None:
    print(f"Binary form: ", binary_result)
    print(f"Decimal form: ", decimal_result)