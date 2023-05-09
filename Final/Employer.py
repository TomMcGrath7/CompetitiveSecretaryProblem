import math
import numpy as np


def probability_best_chosen(n, l, k):
    if k == 1:
        return 1
    if l == 0 and k != 1:
        return 0
    elif (1 < k <= (n - l + 1)):  # confirm the right bound
        numerator = 0
        for i in range(2, k):
            # print(i)
            numerator += (1 / (i - 1)) * ((math.factorial(l) * math.factorial(n - l - 1) * math.factorial(n - i - 1)) /
                                          (math.factorial(l - 1) * math.factorial(n - l - i)))
        denominator = math.factorial(n - 1)
        return numerator / denominator
    elif (k > (n - l + 1)):
        sum = 0
        for i in range(l + 1, n):
            sum += (1 / (i - 1))
        return (l / (n - 1)) * sum

n = 10
l = 3
alpha = 1
k = 6

print(probability_best_chosen(n, l, k))