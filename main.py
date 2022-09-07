from numpy import array, zeros, fabs
import numpy as np
import regex as re
import sys
import copy

def main():
    args = sys.argv[1:]
    if len(args) != 0:
        input_filepath = args[0]
        input_file = open(input_filepath, "r")
        n, a, b = read_file(input_file)
    else:
        n, a, b = read_console()
    x = zeros(n, float)
    c = elimination(a, b, n)
    back_substitution(a, b, n, x)
    d = det(a, n, c)
    r = residual(a, b, n, x)
    small_zero(x, n)

    print("The triangular matrix: \n" + str(a))
    print(b)
    print("The vector of solutions is: \n" + str(x))
    print("The determinant is: \n" + str(d))
    print("The vector of residual is: \n" + str(r))


def to_float_list(x):
    return [float(elem) for elem in x]


def read_file(file):
    n = int(file.readline())
    matrix = file.readlines()
    a_matrix = [re.sub(r'\s+', ' ', line).strip().split(" ") for line in matrix[:n]]
    b_vector = re.sub(r'\s+', ' ', matrix[n]).strip().split(" ")
    a = array([to_float_list(row) for row in a_matrix])
    b = [float(num) for num in b_vector]
    return n, a, b


def read_console():
    n = 0
    while n <= 0 or n > 20:
        try:
            print("Enter value for n: ")
            n = int(input())
            if n <= 0 or n > 20:
                print("N value must be bigger than 0 and less or equal 20!")
        except ValueError:
            print("N value must be number!")
    print("Enter a matrix: ")
    a = zeros((n, n), float)
    i = 0
    while i < n:
        try:
            s = input()
            row = re.sub(r'\s+', ' ', s).strip().split(" ")
            a[i] = [float(val) for val in row]
        except ValueError:
            print("Invalid arguments")
            i -= 1
        i += 1
    print("Enter b values: ")
    b = zeros(n, float)
    try:
        row = re.sub(r'\s+', ' ', input()).strip().split(" ")
        b = [float(val) for val in row]
    except ValueError:
        print("Invalid arguments")
    return n, a, b


# Elimination
def elimination(a, b, n):
    count = 0
    for k in range(n - 1):
        count += partial_pivoting(a, b, k, n)
        for i in range(k + 1, n):
            if a[i, k] == 0: continue
            factor = a[i, k] / a[k, k]
            for j in range(k, n):
                a[i, j] = a[i, j] - a[k, j] * factor
            b[i] = b[i] - b[k] * factor
    return count


# Partial pivoting
def partial_pivoting(a, b, k, n):
    count = 0
    for i in range(k + 1, n):
        if fabs(a[i, k]) > fabs(a[k, k]):
            a[[k, i]] = a[[i, k]]
            temp = b[i]
            b[i] = b[k]
            b[k] = temp
            count += 1
    return count


# Back-substitution
def back_substitution(a, b, n, x):
    x[n - 1] = b[n - 1] / a[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        sum_ax = 0
        for j in range(i + 1, n):
            sum_ax += a[i, j] * x[j]
        x[i] = (b[i] - sum_ax) / a[i, i]


def det(a, n, c):
    product = 1.0 * (-1.0) ** c
    for i in range(n):
        product *= a[i, i]
    return product


def residual(a, b, n, x):
    am = copy.deepcopy(a)
    for i in range(n):
        for j in range(n):
            am[j, i] *= x[i]
    r = zeros(n, float)
    for i in range(n):
        r[i] = fabs(b[i] - np.sum(am[i]))
    return r


def small_zero(x, n):
    for i in range(n):
        if -1.0e-12 <= x[i] <= 1.0e-12:
            x[i] = 0.0


main()
