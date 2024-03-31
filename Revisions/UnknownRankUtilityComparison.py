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


def c_star_n_minus_1(l, n):
    # calculates the probability of a picking the best for a given l and n in the classic game where we can ignore
    # the last player, c*(n-1) would be the l that maximises this expression
    # If l is 0 then there is just a 1/n chance of picking the best. Otherwise we just do the summation from l+1 to n-1.
    # Since we can basically ignore the last player.
    if n == 1:
        return 1
    elif l == 0:
        return 1 / (n-1)
    else:
        return (l / (n - 1)) * sum(1 / (i - 1) for i in range(l + 1, n))

# print(sum_expression_np(n, l))
# print(sum_expression_math(n, l))
# print(n*l/(n-1))

import matplotlib.pyplot as plt

n_values = range(2, 15)
sum_expression_values = []
n_l_div_n_minus_1_values = []

for n in n_values:
    best_c_star_n_minus_1_value = 0
    current_best_l_c_star_n_minus_1 = 0
    print("n is ")
    print(n)
    print("l is ")
    for a in range(0, n):
        # print("a is")
        # print(a)
        if c_star_n_minus_1(a, n) > best_c_star_n_minus_1_value:
            best_c_star_n_minus_1_value = c_star_n_minus_1(a, n)
            current_best_l_c_star_n_minus_1 = a
    l = current_best_l_c_star_n_minus_1
    print(l)
    sum_expression_values.append(sum_expression_np(n, l))
    n_l_div_n_minus_1_values.append(n * l / (n - 1))

plt.figure(figsize=(10, 6))

# Plotting sum_expression_np(n, l) in blue with dots
plt.scatter(n_values, sum_expression_values, color='blue', label='sum_expression_np(n, l)')

# Plotting n*l/(n-1) in red with dots
plt.scatter(n_values, n_l_div_n_minus_1_values, color='red', label='n*l/(n-1)')

plt.xlabel('n')
plt.ylabel('Value')
plt.title('Dot Plot of the two functions')
plt.legend()
plt.grid(True)
plt.show()