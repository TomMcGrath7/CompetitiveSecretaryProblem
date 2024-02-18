import numpy as np
import itertools


def create_custom_permutations(number_of_players, k, l, offset):
    """
    Generate custom permutations of a sequence of players, with one player inserted at a specified position.

    Args:
    - number_of_players (int): Total number of players, determining the set of numbers to permute.
    - k (int): The 1-based index of the player to insert at a specific position in each permutation.
    - l (int): The base 0-based position in the permutation where the fixed player will be inserted after.
    - offset (int): An adjustment to the insertion position of the fixed player. When offset is 0, the player
                    is inserted directly at position l, effectively making it l+1 in human-readable form.

    Returns:
    - list of tuples: A list of permutations, each with the fixed player inserted at the position determined by
                      `l` and adjusted by `offset`.

    This function first creates permutations of numbers from 1 to `number_of_players`, excluding the player `k`.
    Then, for each permutation, it inserts the player `k` at the position `l + offset`, accommodating the player
    at `l+1` position when `offset` is 0, due to the 0-based indexing of Python lists.
    """
    candidates = np.array(list(range(1, number_of_players + 1)))
    fixed_player = candidates[k - 1]
    candidates = np.delete(candidates, k - 1)
    perms = list(itertools.permutations(candidates))
    perms = [perm[:l + offset] + (fixed_player,) + perm[l + offset:] for perm in perms]
    return perms


def classic_stopping_rule_empirical_wins(permutations, number_of_players, l):
    """
    Calculate the empirical win rates for each player based on a set of permutations.

    Args:
    - permutations (list of tuples): A list of permutations representing possible outcomes.
    - number_of_players (int): The total number of players involved in the game.
    - l (int): A specific position that divides each permutation into two segments for analysis.

    Returns:
    - numpy.ndarray: An array of win percentages for each player, indicating the proportion of permutations
                     where each player is considered to have won.

    The function operates by dividing each permutation into two segments at position `l` and determining wins
    based on the lowest player index found before and after this position. A win is assigned to a player if they
    have the lowest index in the post-l segment that is lower than any index in the pre-l segment, or if no such
    player exists, to the player in the last position of the permutation as a default win condition. The win
    percentages are calculated as the number of wins for each player divided by the total number of permutations.
    """
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


