import numpy as np
import math
import itertools


def probability_best_chosen_BR(n, l, k):
    if k == 1:
        return 1
    if l == 0 and k != 1:
        return 0
    elif (l <= (math.factorial(n-l-1)*math.factorial(n-k))/(math.factorial(n-2)*math.factorial(n-k-l))):  # confirm the right bound
        numerator = 0
        for i in range(2, k):
            numerator += (1 / (i - 1)) * ((math.factorial(l) * math.factorial(n - l - 1) * math.factorial(n - i - 1)) /
                                          (math.factorial(l - 1) * math.factorial(n - l - i)))
        denominator = math.factorial(n - 1)
        return numerator / denominator
    elif (l > (math.factorial(n-l-1)*math.factorial(n-k))/(math.factorial(n-2)*math.factorial(n-k-l))):
        sum = 0
        for i in range(l + 1, n):
            sum += (1 / (i - 1))
        return (l / (n - 1)) * sum


def best_l(n):
    l_probs = np.zeros(n-1)
    for l in range(0,n):
        sum = 0
        for k in range(1,n+1):
            sum += probability_best_chosen_BR(n, l, k)
        l_probs[l] = sum
    return l_probs


n = 4
probs = best_l(n)
print(probs)
# max value of probs
max_value = np.max(probs)

print(np.argmax(probs))
print(np.max(probs))
bestL = np.where(probs == max_value)