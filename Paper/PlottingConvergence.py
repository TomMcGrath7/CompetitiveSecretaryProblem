import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb, factorial


# Function Definitions
def ue_prod(l, k, n):
    if k == 1:
        return 1
    elif k >= 2 and l == 0:
        return 0
    elif k >= 2 and 0 < l < np.prod([n - l - i for i in range(1, k)]) / np.prod([n - i for i in range(2, k)]):
        summation = 0
        for i in range(2, k):
            summation += comb(n - (i + 1), l - 1) / (i - 1)
        return (factorial(l) * factorial(n - l - 1) / factorial(n - 1)) * summation
    elif k >= 2:
        summation = 0
        for i in range(l + 1, n):
            summation += 1 / (i - 1)
        return (l / (n - 1)) * summation
    return 0


def ue_fact(l, k, n):
    if k == 1:
        return 1
    elif k != 1 and l == 0:
        return 0
    threshold = factorial(n - l - 1) * factorial(n - k) / (factorial(n - 2) * factorial(n - k - l))
    if k >= (n - l + 1):
        summation = 0
        for i in range(l + 1, n):
            summation += 1 / (i - 1)
        return (l / n) * summation
    elif l < threshold:
        summation = 0
        for i in range(2, k):
            summation += comb(n - (i + 1), l - 1) / (i - 1)
        return (factorial(l) * factorial(n - l - 1) / factorial(n - 1)) * summation
    else:
        summation = 0
        for i in range(l + 1, n):
            summation += 1 / (i - 1)
        return (l / n) * summation


def ue_class(l, n):
    if l == 0:
        return 1 / n
    summation = 0
    for i in range(l + 1, n + 1):
        summation += 1 / (i - 1)
    return (l / n) * summation


def compute_l_star(func, n):
    max_val = -float('inf')
    l_star = 0
    for l in range(n):
        sum_val = sum(func(l, k, n) for k in range(1, n + 1))
        if sum_val > max_val:
            max_val = sum_val
            l_star = l
    return l_star


def compute_l_star_class(func, n):
    max_val = -float('inf')
    l_star = 0
    for l in range(n):
        sum_val = func(l, n)
        if sum_val > max_val:
            max_val = sum_val
            l_star = l
    return l_star


# Compute and Plot
n_range = list(range(1, 151))
l_star_prod_values = [compute_l_star(ue_prod, n) for n in n_range]
l_star_fact_values = [compute_l_star(ue_fact, n) for n in n_range]
l_star_class_values = [compute_l_star_class(ue_class, n) for n in n_range]

plt.figure(figsize=(20, 10))
plt.plot(n_range, [l / n for l, n in zip(l_star_prod_values, n_range)], '-o', label='$\ell^*_{prod}/n$', color='blue',
         markersize=4)
plt.plot(n_range, [l / n for l, n in zip(l_star_fact_values, n_range)], '-o', label='$\ell^*_{fact}/n$', color='red',
         markersize=4)
plt.plot(n_range, [l / n for l, n in zip(l_star_class_values, n_range)], '-o', label='$\ell^*_{class}/n$',
         color='purple', markersize=4)
plt.axhline(y=1 / np.e, color='green', linestyle='--', label='$1/e$')
plt.xlabel('n')
plt.ylabel('$\ell^*/n$')
plt.title('$\ell^*/n$ vs. n for Integer Values of n')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xticks(np.arange(min(n_range), max(n_range) + 1, 5.0))
plt.show()
