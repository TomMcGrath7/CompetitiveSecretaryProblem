import math
import numpy as np
# import matplotlib.pyplot
import pandas as pd

""" Classic Secretary Problem Probabilities """


def classic_probability(n, l):
    if l == 0:
        return 1 / n
    else:
        sum = 0
        for i in range(l + 1, n + 1):
            sum += (1 / (i - 1))
        return (l / n) * sum


""" Probability of candidate getting chosen"""

# Probability of candidate getting chosen without a chosen alpha
# this would be nice as we can then show how much candidates gain from the ability to be strategic

""" Probability of candidate getting chosen with a chosen alpha"""
" Probability they are chosen when choosing alpha or choosing to go at back"


def probability_candidate_chosen(n, l, alpha):
    payoffs = np.empty((n, 2))
    for k in range(1, n + 1):
        if (n - k - l - alpha + 1 >= 0):
            upper = (math.factorial(l) * math.factorial(alpha - 1) * math.factorial(n - l - alpha) * math.factorial(
                n - k))
            lower = (math.factorial(n - 1) * math.factorial(n - k - l - alpha + 1) * math.factorial(l + alpha - 1))
            payoffs[k - 1, 0] = upper / lower
            payoffs[k - 1, 1] = (l / (n - 1))
        else:
            payoffs[k - 1, 0] = 0
            payoffs[k - 1, 1] = (l / (n - 1))
    payoffs[(n - l):n, 0] = 0
    return np.round(payoffs, decimals=4)


""" Probability of employer picking the best candidate given alpha = 1 is chosen"""
" This function takes as input "
" number of players n "
" number of candidates to be rejected l"
" The rank of the strategic candidate k"

" This function is split into 3 cases, when k = 1, when k is between 2 and n-l+1 and when k is greater than n-l+1."
" In the first case the probability of choosing the best is 1"
"In the second case the probability of choosing the best is summing all the times the best does not appear in the " \
"first l and comes before all those better than k "
"In the third case the probability of choosing the best is like the regular secretary problem as this candidate k has " \
"no chance of "


def probability_best_chosen(n, l, k):
    if k == 1:
        return 1
    if l == 0 and k != 1:
        return 0
    elif (1 < k <= (n - l + 1)):  # confirm the right bound
        numerator = 0
        for i in range(2, k):
            # print(i)
            numerator += (1 / (i - 1))*((math.factorial(l) * math.factorial(n - l - 1) * math.factorial(n - i - 1)) /
                (math.factorial(l - 1) * math.factorial(n - l - i)))
        denominator = math.factorial(n - 1)
        return numerator / denominator
    elif (k > (n - l + 1)):
        sum = 0
        for i in range(l + 1, n):
            sum += (1 / (i - 1))
        return (l / (n - 1)) * sum


n = 6
l = 1
alpha = (6-3)
k = 6

print(probability_best_chosen(n, l, k))

""" Getting robust l """
n = 4
# Define array of length n-1
probs = np.zeros(n-1)

for l in range(0, n-1):
    sum = 0
    for k in range(1, n-l+1):
        sum += probability_best_chosen(n, l, k)
    probs[l] = sum

print(probs)
# find max value and index of probs
max_value = np.amax(probs)
print(max_value)
# what is index of max value
max_index = np.where(probs == max_value)
print(max_index)

