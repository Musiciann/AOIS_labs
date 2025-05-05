from constants import *

def decimal_to_binary_direct_code(decimal_num: int) -> str:

    if decimal_num == 0:
        return VALUE_ZERO

    binary_num = ""

    is_negative = decimal_num < 0
    decimal_num = abs(decimal_num)

    while decimal_num > 0:
        modulo = decimal_num % 2
        binary_num = str(modulo) + binary_num
        decimal_num //= 2

    binary_num = binary_num.zfill(7)

    return (VALUE_ONE if is_negative else VALUE_ZERO) + binary_num


def decimal_to_binary_inverse_code(decimal_num: int) -> str:
    binary_inverse = decimal_to_binary_direct_code(decimal_num)

    if decimal_num >= 0:
        return binary_inverse

    binary_inverse = list(binary_inverse)
    for i in range(1, len(binary_inverse)):
        if binary_inverse[i] == VALUE_ZERO:
            binary_inverse[i] = VALUE_ONE
        else:
            binary_inverse[i] = VALUE_ZERO

    return "".join(binary_inverse)

def decimal_to_binary_additional_code(decimal_num: int) -> str:
    if decimal_num >= 0:
        binary_additional = decimal_to_binary_direct_code(decimal_num)
        return binary_additional
    else:
        binary_additional = decimal_to_binary_inverse_code(decimal_num)
        binary_additional = additional_code_binary_sum(binary_additional, VALUE_ONE_BINARY)
        return binary_additional

def additional_code_binary_sum(first_binary: str, second_binary: str) -> str:
    carry_num = 0
    result_binary = ""
    for i in range(7, -1, -1):
        total = carry_num + (1 if list(first_binary)[i] == VALUE_ONE else 0) + (1 if list(second_binary)[i] == VALUE_ONE else 0)
        result_binary = str(total % 2) + result_binary
        carry_num = total // 2

    if carry_num:
        result_binary = VALUE_ONE + result_binary

    if len(result_binary) > 8:
        result_binary = result_binary[1:]

    return result_binary

def binary_to_decimal_direct_code(binary_num: str) -> int:
    decimal_num = 0
    sign = binary_num[0]

    for i, digit in enumerate(reversed(binary_num[1:])):
        decimal_num += int(digit) * (2 ** i)

    if sign == VALUE_ONE:
        decimal_num *= -1

    return decimal_num

def binary_to_decimal_fract(binary_num: str) -> int:
    decimal_fract = 0

    for i, digit in enumerate(binary_num[0:]):
        decimal_fract += float(float(digit) * (2 ** (-1 * (i + 1 ))))

    return decimal_fract

def binary_to_decimal_additional_code(additional_binary_num: str) -> int:

    sign = additional_binary_num[0]

    if sign == VALUE_ZERO:
        return binary_to_decimal_direct_code(additional_binary_num)
    else:
        additional_binary_num = list(additional_binary_num)
        for i in range(1, len(additional_binary_num)):
            if additional_binary_num[i] == VALUE_ZERO:
                additional_binary_num[i] = VALUE_ONE
            else:
                additional_binary_num[i] = VALUE_ZERO
        additional_binary_num = "".join(additional_binary_num)

        if sign == VALUE_ONE:
            additional_binary_num = additional_code_binary_sum(additional_binary_num, VALUE_ONE_BINARY)

        decimal_num = binary_to_decimal_direct_code(additional_binary_num)

    return decimal_num

def binary_multiply(first_binary: str, second_binary: str) -> str:

    sign_value = VALUE_ONE if first_binary[0] != second_binary[0] else VALUE_ZERO

    first_binary = str(int(first_binary[1:]))
    second_binary = str(int(second_binary[1:]))

    result = VALUE_ZERO * (len(first_binary) + len(second_binary))
    shift = 0
    for i in range(len(first_binary) - 1, -1, -1):
        if first_binary[i] == VALUE_ONE:
            shifted_second_binary = second_binary + VALUE_ZERO * shift
            result = binary_add_shifted(result, shifted_second_binary)
        shift += 1

    result_value = result[-7:].zfill(7)

    return sign_value + result_value

def binary_add_shifted(first_binary: str, second_binary: str) -> str:
    max_len = max(len(first_binary), len(second_binary))
    first_binary = first_binary.zfill(max_len)
    second_binary = second_binary.zfill(max_len)

    carry_num = 0
    result_binary = []

    for i in range(max_len - 1, -1, -1):
        total = carry_num + int(first_binary[i]) + int(second_binary[i])
        result_binary.append(str(total % 2))
        carry_num = total // 2

    if carry_num:
        result_binary.append(VALUE_ONE)
    return ''.join(result_binary[::-1])

def binary_divide(dividend: str, divisor: str, precision: int = 8) -> str:
    sign = determine_sign(dividend, divisor)
    dividend_num, divisor_num = prepare_numbers(dividend, divisor)

    check_division_by_zero(divisor_num)
    if is_zero(dividend_num):
        return sign + VALUE_ZERO

    dividend_num, divisor_num = normalize_lengths(dividend_num, divisor_num)

    int_part, remainder = perform_integer_division(dividend_num, divisor_num)
    frac_part = perform_fractional_division(remainder, divisor_num, precision)

    return format_result(sign, int_part, frac_part)

def determine_sign(dividend: str, divisor: str) -> str:
    return VALUE_ONE if dividend[0] != divisor[0] else VALUE_ZERO

def prepare_numbers(dividend: str, divisor: str) -> tuple[str, str]:
    dividend_num = dividend[1:].lstrip(VALUE_ZERO) or VALUE_ZERO
    divisor_num = divisor[1:].lstrip(VALUE_ZERO) or VALUE_ZERO
    return dividend_num, divisor_num

def check_division_by_zero(divisor: str) -> None:
    if divisor == VALUE_ZERO:
        raise ValueError("ValueError: Division By Zero!")

def is_zero(number: str) -> bool:
    return number == VALUE_ZERO

def normalize_lengths(a: str, b: str) -> tuple[str, str]:
    max_len = max(len(a), len(b))
    return a.zfill(max_len), b.zfill(max_len)

def perform_integer_division(dividend: str, divisor: str) -> tuple[str, str]:
    quotient = []
    remainder = VALUE_ZERO

    for bit in dividend:
        remainder = shift_left(remainder, bit)
        if binary_compare(remainder, divisor) >= 0:
            remainder = binary_subtract(remainder, divisor)
            quotient.append(VALUE_ONE)
        else:
            quotient.append(VALUE_ZERO)

    int_part = ''.join(quotient).lstrip(VALUE_ZERO) or VALUE_ZERO
    return int_part, remainder

def perform_fractional_division(remainder: str, divisor: str, precision: int) -> str:
    frac_part = []

    for _ in range(precision):
        if is_zero(remainder):
            frac_part.append(VALUE_ZERO)
            continue

        remainder = shift_left(remainder, VALUE_ZERO)
        if binary_compare(remainder, divisor) >= 0:
            remainder = binary_subtract(remainder, divisor)
            frac_part.append(VALUE_ONE)
        else:
            frac_part.append(VALUE_ZERO)

    return ''.join(frac_part).rstrip(VALUE_ZERO)

def shift_left(number: str, new_bit: str) -> str:
    return binary_add(number + new_bit, '0')

def format_result(sign: str, int_part: str, frac_part: str) -> str:
    result = sign + int_part
    if frac_part:
        result += '.' + frac_part
    return result

def binary_compare(a: str, b: str) -> int:
    a = a.lstrip(VALUE_ZERO) or VALUE_ZERO
    b = b.lstrip(VALUE_ZERO) or VALUE_ZERO

    if len(a) > len(b):
        return 1
    elif len(a) < len(b):
        return -1

    for bit_a, bit_b in zip(a, b):
        if bit_a > bit_b:
            return 1
        elif bit_a < bit_b:
            return -1

    return 0


def binary_add(a: str, b: str) -> str:
    max_len = max(len(a), len(b))
    a = a.zfill(max_len)
    b = b.zfill(max_len)

    carry = 0
    result = []

    for i in range(max_len - 1, -1, -1):
        sum_bits = carry + int(a[i]) + int(b[i])
        result.append(str(sum_bits % 2))
        carry = sum_bits // 2

    if carry:
        result.append(VALUE_ONE)

    return ''.join(reversed(result))


def binary_subtract(a: str, b: str) -> str:
    max_len = max(len(a), len(b))
    a = a.zfill(max_len)
    b = b.zfill(max_len)

    borrow = 0
    result = []

    for i in range(max_len - 1, -1, -1):
        a_bit = int(a[i]) - borrow
        b_bit = int(b[i])

        if a_bit < b_bit:
            a_bit += 2
            borrow = 1
        else:
            borrow = 0

        result.append(str(a_bit - b_bit))

    while len(result) > 1 and result[-1] == VALUE_ZERO:
        result.pop()

    return ''.join(reversed(result))

def binary_to_decimal_division(binary_value: str) -> float:
    sign_value = 1 if binary_value[0] == VALUE_ZERO else -1
    binary_value = binary_value[1:]
    integer_binary, fract_binary = binary_value.split('.')
    integer_decimal = binary_to_decimal_direct_code(integer_binary.zfill(8))
    fract_decimal = binary_to_decimal_fract(fract_binary)
    return sign_value * (integer_decimal + fract_decimal)

def float_to_ieee754(num):
    sign_bit = VALUE_ONE if num < 0 else VALUE_ZERO
    if num == 0:
        return VALUE_ZERO * 32

    num = abs(num)
    integer_part = int(num)
    fractional_part = num - integer_part

    integer_binary = bin(integer_part)[2:] if integer_part > 0 else ''
    fractional_binary = ''

    while fractional_part and len(fractional_binary) < 23:
        fractional_part *= 2
        bit = int(fractional_part)
        fractional_binary += str(bit)
        fractional_part -= bit

    if integer_binary:
        exponent = len(integer_binary) - 1
    elif VALUE_ONE in fractional_binary:
        exponent = -fractional_binary.index(VALUE_ONE) - 1
    else:
        return VALUE_ZERO * 32

    exponent_bits = bin(exponent + 127)[2:].zfill(8)
    mantissa_bits = (integer_binary[1:] + fractional_binary).ljust(23, '0')[:23]

    return f'{sign_bit}{exponent_bits}{mantissa_bits}'


def ieee754_to_float(ieee_binary):
    sign = int(ieee_binary[0])
    exponent = int(ieee_binary[1:9], 2) - 127
    mantissa = ieee_binary[9:]

    if exponent == -127 and all(b == VALUE_ZERO for b in mantissa):
        return 0.0

    mantissa_value = 1

    for i, bit in enumerate(mantissa):
        if bit == VALUE_ONE:
            mantissa_value += 2 ** -(i + 1)

    return (-1) ** sign * mantissa_value * (2 ** exponent)

def sum_floats_ieee754(first_float, second_float):
    if first_float < 0 or second_float < 0:
        raise ValueError("Error Sign Value.")
    first_binary = float_to_ieee754(first_float)
    second_binary = float_to_ieee754(second_float)
    first_exponent = int(first_binary[1:9], 2)
    first_mantissa = int(first_binary[9:], 2)
    second_exponent = int(second_binary[1:9], 2)
    second_mantissa = int(second_binary[9:], 2)

    first_mantissa |= (1 << 23)
    second_mantissa |= (1 << 23)

    if first_exponent > second_exponent:
        second_mantissa >>= (first_exponent - second_exponent)
        exponent = first_exponent
    else:
        first_mantissa >>= (second_exponent - first_exponent)
        exponent = second_exponent

    result_mantissa = first_mantissa + second_mantissa
    if result_mantissa & (1 << 24):
        result_mantissa >>= 1
        exponent += 1
    result_mantissa &= ~(1 << 23)

    if exponent >= 255:
        raise OverflowError("Exponent Overflow.")

    result_binary = f"{0:01b}{format(exponent, '08b')}{format(result_mantissa, '023b')}"
    result_float = ieee754_to_float(result_binary)

    return result_float, result_binary





