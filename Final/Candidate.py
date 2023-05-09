import math
import numpy as np
import matplotlib.pyplot
import pandas as pd
import itertools


# Document the function beginning on line 8
# This function takes in three parameters: n, l, and alpha
# n is the number of candidates
# l is the number of candidates that the employer will reject no matter what
# alpha is the position that the candidate can position himself in
# Usually alpha is 1, but it can be any number between 1 and n-l
# The output of this function is a 2D array of size n x 2
# The first column is the expected payoff of the candidate if he chooses alpha
# The second column is the probability that the candidate will be chosen if he chooses the final position

def expected_candidate_payoffs(n, l, alpha):
    payoffs = np.empty((n, 2))
    for k in range(1, n + 1):
        if n - k - l - alpha + 2 > 0:
            upper = (math.factorial(l) * math.factorial(alpha - 1) * math.factorial(n - l - alpha) * math.factorial(
                n - k))
            lower = (math.factorial(n - 1) * math.factorial(n - k - l - alpha + 1) * math.factorial(l + alpha - 1))
            payoffs[k - 1, 0] = upper / lower
            payoffs[k - 1, 1] = (l / (n - 1))
        else:
            payoffs[k - 1, 0] = 0
            payoffs[k - 1, 1] = (l / (n - 1))

    # payoffs[(n - l):n, 0] = 0

    return np.round(payoffs, decimals=4)


n = 10
l = 3
alpha = 1

expected_payoff = expected_candidate_payoffs(n, l, alpha)
print(expected_payoff)

""" Printing nice tables"""
index = []
for a in range(0, n):
    current_k = str(a + 1)
    index.append("k = " + current_k)

df = pd.DataFrame(expected_payoff, columns=['alpha = 1', 'alpha = (n-l)'], index=index)
# print(df)
print(df.to_latex(index=True))


def amount_preferring_alpha1(payoffs):
    amount = 0
    for i in range(0, len(payoffs)):
        if payoffs[i, 0] > payoffs[i, 1]:
            amount += 1
    return amount


print(amount_preferring_alpha1(expected_payoff))
