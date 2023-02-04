import math
import numpy as np
import matplotlib.pyplot

e = math.e

n = 10
l = int(round(n/e, 0))
print(l)

payoffs1 = np.empty((n, 2))
payoffs = np.empty((n, 2))

# for i in range(1, n):
#     payoffs1[i-1, 0] = (1-(l/n))**(i-1)
#     payoffs1[i-1, 1] = (l/n)

# print(list(range(n)))

payoffs[0:n, 0] = (1-(l/n))**(np.array(list(range(n))))
payoffs[0:n, 1] = (l/n)
payoffs[(n-l):n, 0] = 0


print(np.round(payoffs, decimals=3))

# print(np.round(payoffs1, decimals=3))


def prob_best(x):
    return -x*np.log(x)

# print(prob_best(.5))
""" Pass an array that goes from 0 to 1 to prob_best function and plot"""
# 1/e maximises the function

# Write Program to compute best l for given sample size n



