from Minimization import LogicalFunctionMinimization

#(a>b)&!c|(d~e)
#(a&b)|!c
#!(!a>!b)|c

if __name__ == '__main__':

    formula = input("Enter a logical formula >>> ")
    lf = LogicalFunctionMinimization(formula)
    lf.generate_truth_table()
    print('СДНФ:', lf.minimize_sdnf_quine(), '\n')
    print('СКНФ:', lf.minimize_sknf_quine(), '\n\n\n')
    print(lf.minimize_sdnf_calculational(), '\n')
    print(lf.minimize_sknf_calculational(), '\n\n\n')
    print(lf.minimize_sdnf_karnaugh(), '\n')
    print(lf.minimize_sknf_karnaugh())