from LF import LogicalFunction
from Print_Truth_Table_Function import print_truth_table, show_truth_table_result

if __name__ == "__main__":
    expr = input("Insert expression >>> ")
    logical_func = LogicalFunction(expr)
    truth_table, variables, expression = logical_func.generate_truth_table()
    print_truth_table(truth_table, variables, expression)
    binary_result, decimal_result = logical_func.truth_table_result()
    show_truth_table_result(binary_result, decimal_result)
    print("SDNF:", logical_func.generate_sdnf())
    print("SDNF numeric:", logical_func.generate_sdnf_numeric())
    print("SCNF:", logical_func.generate_scnf())
    print("SCNF numeric:", logical_func.generate_scnf_numeric())