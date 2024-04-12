import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

# Function to calculate the given term for n and different l's
def calculate_term(n, l):
    term = 0
    for i in range(7):
        if n-i-1-l > 0:
            term += factorial(n-i-1) / factorial(n-i-1-l)
    if n-7-l > 0:
        term += (n-l-6) * (factorial(n-7) / factorial(n-7-l))
    return factorial(n-l-1) / factorial(n-2) * term

n = 10  # Keeping n constant
l_values = np.arange(0, n-1)  # l values from 0 to n-2
terms = [calculate_term(n, l) for l in l_values]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(l_values, terms, marker='o', linestyle='-', color='blue')
plt.title(f'Plot of the term for n={n} and different l\'s')
plt.xlabel('l values')
plt.ylabel('Term value')
plt.grid(True)
plt.show()
