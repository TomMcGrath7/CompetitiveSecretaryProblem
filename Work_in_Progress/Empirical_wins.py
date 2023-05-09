import itertools
import numpy as np


# This function returns the empirical number of times somebody wins without any choice
# Basically if you were to have every possible permutation, what are the probabilities that each person wins

def empirical_wins_without_choice(n, l):
    candidates = np.array(list(range(1, n + 1)))
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

emp = empirical_wins_without_choice(n, l)
print(emp)
