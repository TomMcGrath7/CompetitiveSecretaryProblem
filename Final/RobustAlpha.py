import math
import numpy as np


def payoff(n, l, alpha, k):
    if alpha == 1 and k < (n - l + 1):
        return (math.factorial(l) * math.factorial(n - l - 1) * math.factorial(n - k)) / (
                    math.factorial(n - 1) * math.factorial(l) * math.factorial(n - k - l))
    elif alpha == 1 and k >= (n - l + 1):
        return 0
    elif alpha == (n - l):
        return l / (n - 1)


# test = payoff(10, 3, 7, 2)
# print(test)


def robust_alpha(n_max, l):
    output = np.zeros((n_max + 1, 3))
    for n in range(1, n_max + 1):
        alpha_1 = 0
        alpha_n_l = 0
        for k in range(1, n + 1):
            alpha_1 += payoff(n, l, 1, k)
            alpha_n_l += payoff(n, l, n - l, k)

        output[n, 0] = n
        if alpha_1 > alpha_n_l:
            output[n, 1] = 1
            output[n, 2] = alpha_1 / n
        elif alpha_1 <= alpha_n_l:
            output[n, 1] = n - l
            output[n, 2] = alpha_n_l / n

    return output



e = math.e
# print(e)
n_max = 100
l = round(n_max/e)
# print(l)


