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


n = 6
l = 3
alpha = (6-3)
k = 6

emp = empirical_wins_without_choice(n, l)
print(emp)


def create_custom_permutations(number_of_players, k, l, alpha):
    candidates = np.array(list(range(1, number_of_players + 1)))
    fixed_player = candidates[k - 1]
    candidates = np.delete(candidates, k - 1)
    perms = list(itertools.permutations(candidates))
    perms = [perm[:l + alpha - 1] + (fixed_player,) + perm[l + alpha - 1:] for perm in perms]
    return perms


perms = create_custom_permutations(n, k, l, alpha)
print(perms)


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


empirical_wins_with_choice = empirical_wins(perms, n, l)

print(empirical_wins_with_choice)
