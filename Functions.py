import numpy as np
import itertools


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