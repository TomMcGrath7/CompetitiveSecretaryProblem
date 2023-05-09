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
            upper = (math.factorial(l) * math.factorial(alpha - 1) * math.factorial(n - l - alpha) * math.factorial(n - k))
            lower = (math.factorial(n - 1) * math.factorial(n - k - l - alpha + 1) * math.factorial(l + alpha - 1))
            payoffs[k - 1, 0] = upper/lower
            payoffs[k - 1, 1] = (l / (n - 1))
        else:
            payoffs[k - 1, 0] = 0
            payoffs[k - 1, 1] = (l / (n - 1))

    # payoffs[(n - l):n, 0] = 0

    return np.round(payoffs, decimals=4)


def empirical_wins_without_choice(n, l):
    candidates = np.array(list(range(1, n+1)))
    perms = list(itertools.permutations(candidates))
    wins = np.zeros(n)
    for i in range(len(perms)):
        best_rank = n
        for j in range(0, l):
            if perms[i][j] < best_rank:
                best_rank = perms[i][j]
        best_l = best_rank
        for k in range(l, n):
            if perms[i][k] < best_rank:
                best_rank = perms[i][k]
                wins[best_rank - 1] += 1
                break
        if best_l == best_rank:
            wins[perms[i][n - 1] - 1] += 1

    win_percents = wins / len(perms)
    return win_percents

n = 10
l = 3
alpha = 1

expected_payoff = expected_candidate_payoffs(n, l, alpha)
print(expected_payoff)

emp = empirical_wins_without_choice(n, l)
print(emp)









