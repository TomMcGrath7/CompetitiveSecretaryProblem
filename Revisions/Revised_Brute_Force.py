import numpy as np
import itertools


def create_custom_permutations(number_of_players, k, alpha):
    """
    Generate custom permutations of a sequence of players, inserting one player at their preferred position.

    Args:
    - number_of_players (int): Total number of players, determining the set of numbers to permute.
    - k (int): The 1-based index of the player to be inserted at a specific preferred position in each permutation.
    - alpha (int): The 1-based preferred position of player `k` within the permutation, enabling dynamic insertion based on player preference.

    Returns:
    - list of tuples: A list of permutations, each with player `k` inserted at the position determined by `alpha`.

    This function creates permutations of numbers from 1 to `number_of_players` excluding `k`. It then inserts `k` back into these permutations at a position that aligns with `k`'s preference (`alpha`). The function allows for flexible positioning of player `k` within the permutations, enhancing strategic placement in various game scenarios.
    """
    candidates = np.array(list(range(1, number_of_players + 1)))
    fixed_player = candidates[k - 1]
    candidates = np.delete(candidates, k - 1)
    perms = list(itertools.permutations(candidates))
    adjusted_position = alpha - 1  # Convert alpha to 0-based index for insertion
    perms = [perm[:adjusted_position] + (fixed_player,) + perm[adjusted_position:] for perm in perms]
    return perms


def classic_stopping_rule_empirical_wins(permutations, l):
    """
    Calculate the empirical win rates for each player based on a set of permutations, including consideration
    for the employer's win condition tied to the selection of the player with the best rank (k=1).

    Args:
    - permutations (list of tuples): A list of permutations representing possible outcomes.
    - l (int): A specific position that divides each permutation into two segments for analysis.

    Returns:
    - numpy.ndarray: An array of win percentages for each player, indicating the proportion of permutations
                     where each player is considered to have won. The employer's win rate is conceptually
                     equivalent to the win rate of the player with rank k=1, as selecting the top-ranked player
                     is considered a win for the employer.

    The function divides each permutation into two segments at position `l` and determines wins based on the
    lowest player index found before and after this position. A win is assigned to a player if they have the
    lowest index in the post-l segment that is lower than any index in the pre-l segment. If no such player
    exists, the win defaults to the player in the last position. The win rate for the employer is implicitly
    represented by the win rate of the player with rank k=1, without the need for separate tracking.
    """
    if not permutations:
        return np.array([])  # Return an empty array if permutations list is empty

    number_of_players = len(permutations[0])  # Determine the number of players from the length of a permutation
    wins = np.zeros(number_of_players)
    for perm in permutations:
        best_rank = number_of_players + 1
        for j in range(0, l):
            if perm[j] < best_rank:
                best_rank = perm[j]
        best_l = best_rank
        for k in range(l, number_of_players):
            if perm[k] < best_rank:
                best_rank = perm[k]
                wins[best_rank - 1] += 1
                break
        if best_l == best_rank:
            wins[perm[number_of_players - 1] - 1] += 1

    win_percents = wins / len(permutations)
    return win_percents


n = 10
l = 3
alpha = 10
k = 1

perms = create_custom_permutations(n, k, alpha)
emp_wins = classic_stopping_rule_empirical_wins(perms, l)
print(emp_wins)