import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

n = 100
ell_switches = []

for k in range(1, n):
    ell_prev = None
    for ell in range(n-1):
        lhs = ell
        rhs = factorial(n - ell - 1) * factorial(n - k) / (factorial(n - 2) * factorial(n - k - ell))
        if rhs < lhs:
            ell_prev = ell
            break
    ell_switches.append(ell_prev)

plt.plot(range(1, n), ell_switches)
plt.xlabel('k')
plt.ylabel('ell where the inequality switches')
plt.title('Value of ell where the inequality switches for different k')
plt.grid(True)
plt.show()