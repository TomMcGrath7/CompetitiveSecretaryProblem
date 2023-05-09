import numpy as np


def prob_best(x):
    return -x * np.log(x)


""" Classic Secretary Problem Probabilities """


def classic_probability(n, l):
    if l == 0:
        return 1 / n
    else:
        sum = 0
        for i in range(l + 1, n + 1):
            sum += (1 / (i - 1))
        return (l / n) * sum
