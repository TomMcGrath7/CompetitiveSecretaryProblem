import numpy as np
import itertools
import math

""" Functions used """

def create_custom_permutations(number_of_players, k, l, alpha):
    candidates = np.array(list(range(1, number_of_players + 1)))
    fixed_player = candidates[k-1]
    candidates = np.delete(candidates, k - 1)
    perms = list(itertools.permutations(candidates))
    perms = [perm[:l + alpha - 1] + (fixed_player,) + perm[l + alpha - 1:] for perm in perms]
    return perms


def empirical_wins(permutations, number_of_players, l):
    wins = np.zeros(number_of_players)
    for i in range(len(permutations)):
        best_rank = number_of_players
        for j in range(0, l):
            if permutations[i][j] < best_rank:
                best_rank = permutations[i][j]
        best_l = best_rank
        for k in range(l, number_of_players):
            if permutations[i][k] < best_rank:
                best_rank = permutations[i][k]
                wins[best_rank - 1] += 1
                break
        if best_l == best_rank:
            wins[permutations[i][number_of_players - 1] - 1] += 1

    win_percents = wins / len(permutations)
    return win_percents


def expected_payoffs(n, l, alpha):
    payoffs = np.empty((n, 2))
    for k in range(1, n + 1):
        if (n - k - l - alpha + 1 > 0):
            upper = (math.factorial(l) * math.factorial(alpha - 1) * math.factorial(n - l - alpha) * math.factorial(n - k))
            lower = (math.factorial(n - 1) * math.factorial(n - k - l - alpha + 1) * math.factorial(l + alpha - 1))
            payoffs[k - 1, 0] = upper/lower
            payoffs[k - 1, 1] = (l / (n - 1))
        else:
            payoffs[k - 1, 0] = 0
            payoffs[k - 1, 1] = (l / (n - 1))

    payoffs[(n - l):n, 0] = 0

    return np.round(payoffs, decimals=3)

""" Inputs """
n = 6
k = 3

""" Find best l """
best_l = 0
best_probability = 0

for l in range(1, n-1):
    print("l is ")
    print(l)
    alpha = 1
    expected_payoff = expected_payoffs(n, l, 1)
    print(expected_payoff)
    print(expected_payoff[k-1][0])
    print(expected_payoff[k-1][1])
    if expected_payoff[k-1][0] >= expected_payoff[k-1][1]:
        alpha = 1
    elif expected_payoff[k-1][0] < expected_payoff[k-1][1]:
        alpha = n-l
    else:
        print("error")
    perms = create_custom_permutations(n, k, l, alpha)
    print(perms)
    win_percents = empirical_wins(perms, n, l)
    print(win_percents)
    if win_percents[0] > best_probability:
        best_probability = win_percents[0]
        best_l = l


print(best_probability)
print(best_l)

