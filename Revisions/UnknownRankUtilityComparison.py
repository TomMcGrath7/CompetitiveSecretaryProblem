import numpy as np
from scipy.special import factorial, comb


def sum_expression_np(n, l):
    if n <= 0 or l < 0 or l >= n:
        return 0  # Handling invalid inputs

    # Creating an array for k values
    k_values = np.arange(1, n + 1)
    # Calculating binomial coefficients for all k in a vectorized manner
    binom_coeffs = comb(n - k_values, l, exact=False)

    # Calculating the constant term outside the sum
    constant_term = factorial(l) * factorial(n - l - 1) / factorial(n - 1)

    # Calculating the total sum
    total_sum = np.sum(constant_term * binom_coeffs)

    return total_sum


import math


def binomial_coefficient(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


def sum_expression_math(n, l):
    # if n <= 0 or l < 0 or l >= n:
    #     return 0  # Handling invalid inputs

    total_sum = 0
    for k in range(1, n + 1):
        if k >= n-l+1:
            total_sum += 0
        else:
            binom_coeff = binomial_coefficient(n - k, l)
            term = math.factorial(l) * math.factorial(n - l - 1) / math.factorial(n - 1)
            total_sum += term * binom_coeff

    return total_sum

n = 2
l = int(np.ceil(n / np.e))
print(sum_expression_np(n, l))
print(sum_expression_math(n, l))
print(n*l/(n-1))

import matplotlib.pyplot as plt

n_values = range(2, 30)
sum_expression_values = []
n_l_div_n_minus_1_values = []

for n in n_values:
    print("n is ")
    print(n)
    print("l is ")
    l = int(np.ceil(n / np.e))
    print(l)
    sum_expression_values.append(sum_expression_np(n, l))
    n_l_div_n_minus_1_values.append(n * l / (n - 1))

plt.figure(figsize=(10, 6))

# Plotting sum_expression_np(n, l) in blue
plt.plot(n_values, sum_expression_values, color='blue', label='sum_expression_np(n, l)')

# Plotting n*l/(n-1) in red
plt.plot(n_values, n_l_div_n_minus_1_values, color='red', label='n*l/(n-1)')

plt.xlabel('n')
plt.ylabel('Value')
plt.title('Plot of the two functions')
plt.legend()
plt.grid(True)
plt.show()