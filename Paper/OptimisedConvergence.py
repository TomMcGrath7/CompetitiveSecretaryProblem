import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext
from scipy.special import comb

# Set precision for decimal operations
getcontext().prec = 100

# Using a cache to store already computed values can save computation time
factorial_cache = {}

def cached_factorial_decimal(n):
    """Computes factorial using caching and Decimal for high precision."""
    if n in factorial_cache:
        return factorial_cache[n]
    if n == 0:
        factorial_cache[0] = Decimal(1)
        return Decimal(1)
    result = Decimal(n) * cached_factorial_decimal(n-1)
    factorial_cache[n] = result
    return result

def comb_decimal(n, k):
    """Computes combinations using Decimal for high precision."""
    return cached_factorial_decimal(n) / (cached_factorial_decimal(k) * cached_factorial_decimal(n-k))

def ue_prod_optimized(l, k, n):
    """Optimized ue_prod function using Decimal for high precision."""
    if k == 1:
        return 1
    elif k >= 2 and l == 0:
        return 0
    elif k >= 2:
        threshold = Decimal(int(np.prod([n - l - i for i in range(1, k)]))) / Decimal(int(np.prod([n - i for i in range(2, k)])))
        if 0 < l < threshold:
            summation = 0
            for i in range(2, k):
                summation += comb_decimal(n - (i + 1), l - 1) / (i - 1)
            return (cached_factorial_decimal(l) * cached_factorial_decimal(n - l - 1) / cached_factorial_decimal(n - 1)) * summation
        else:
            summation = 0
            for i in range(l + 1, n):
                summation += 1 / (i - 1)
            return (l / (n - 1)) * summation
    return 0

def ue_fact_optimized(l, k, n):
    """Optimized ue_fact function using Decimal for high precision."""
    if k == 1:
        return 1
    elif k != 1 and l == 0:
        return 0
    l = int(l)  # Ensure l is an integer scalar
    # Check the k condition first
    if k >= (n - l + 1):
        summation = 0
        for i in range(l + 1, n):
            summation += 1 / (i - 1)
        return (l / n) * summation
    else:
        threshold = cached_factorial_decimal(n - l - 1) * cached_factorial_decimal(n - k) / \
                    (cached_factorial_decimal(n - 2) * cached_factorial_decimal(n - k - l))
        if l < threshold:
            summation = 0
            for i in range(2, k):
                summation += comb_decimal(n - (i + 1), l - 1) / (i - 1)
            return (cached_factorial_decimal(l) * cached_factorial_decimal(n - l - 1) / cached_factorial_decimal(n - 1)) * summation
        else:
            summation = 0
            for i in range(l + 1, n):
                summation += 1 / (i - 1)
            return (l / n) * summation
    return 0

def compute_l_star(func, n):
    """Compute l_star given a function and n."""
    l_values = np.arange(n)
    sum_vals = [np.sum(func(l, np.arange(1, n+1), n)) for l in l_values]
    return l_values[np.argmax(sum_vals)]

def compute_l_star_class(n):
    """Compute l_star for the ue_class function."""
    if n == 1:
        return 0
    l_values = np.arange(1, n)
    sum_vals = [(l / n) * np.sum([1 / (i - 1) for i in range(l + 1, n)]) for l in l_values]
    return np.argmax(sum_vals) + 1  # +1 since l starts from 1 in the computation

# Compute l_star values
n_range = np.arange(1, 226)  # Or change 226 to a larger value if required
l_star_fact_values = [compute_l_star(ue_fact_optimized, n) for n in n_range]
l_star_prod_values = [compute_l_star(ue_prod_optimized, n) for n in n_range]
l_star_class_values = [compute_l_star_class(n) for n in n_range]

# Plot the results
plt.figure(figsize=(22, 11))
plt.plot(n_range, [l/n for l, n in zip(l_star_fact_values, n_range)], '-o', label='$\ell^*_{fact}/n$', color='red', markersize=4)
plt.plot(n_range, [l/n for l, n in zip(l_star_prod_values, n_range)], '-o', label='$\ell^*_{prod}/n$', color='blue', markersize=4)
plt.plot(n_range, [l/n for l, n in zip(l_star_class_values, n_range)], '-o', label='$\ell^*_{class}/n$', color='purple', markersize=4)
plt.axhline(y=1/np.e, color='green', linestyle='--', label='$1/e$')
plt.xlabel('n')
plt.ylabel('$\ell^*/n$')
plt.title('$\ell^*/n$ vs. n')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
