from scipy.special import binom
import numpy as np


# Define the functions to calculate the sums for C* and C*(n-1)

def sum_for_c_star(ell, n):
    if ell == 0:
        return 1 / n
    else:
        return (ell / n) * sum(1 / (i - 1) for i in range(ell + 1, n + 1))


def sum_for_c_star_n_minus_1(ell, n):
    if ell == 0:
        return 1 / n
    else:
        return (ell / (n - 1)) * sum(1 / (i - 1) for i in range(ell + 1, n))


# Calculate the values of C* and C*(n-1) for n from 1 to 20
c_star_values = []
c_star_n_minus_1_values = []

for n in range(1, 21):
    # Find the ell that maximizes the sum for C*
    c_star = max(range(n), key=lambda ell: sum_for_c_star(ell, n), default=0)

    # For C*(n-1), we handle the case when n=1 separately, as it would otherwise result in division by zero
    if n == 1:
        c_star_n_minus_1 = 0
    else:
        c_star_n_minus_1 = max(range(n - 1), key=lambda ell: sum_for_c_star_n_minus_1(ell, n), default=0)

    c_star_values.append(c_star)
    c_star_n_minus_1_values.append(c_star_n_minus_1)

c_star_values, c_star_n_minus_1_values