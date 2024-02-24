import numpy as np
import itertools
import matplotlib.pyplot as plt


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


# n = 10
# l = 3
# alpha = 10
# k = 1
#
# perms = create_custom_permutations(n, k, alpha)
# emp_wins = classic_stopping_rule_empirical_wins(perms, l)
# print(emp_wins)


def revised_stopping_rule_empirical_wins(permutations, l):
    """
    Calculate the empirical win rates for each player based on a set of permutations, under a revised stopping
    rule. A player wins if they are the best seen so far after 'l', surpassing even the first candidate who was
    better than all players before 'l' and subsequently skipped.

    Args:
    - permutations (list of tuples): A list of permutations representing possible outcomes.
    - l (int): A specific position that divides each permutation into two segments for analysis.

    Returns:
    - numpy.ndarray: An array of win percentages for each player, indicating the proportion of permutations
                     where each player is considered to have won under the revised stopping rule.

    Under this rule, the function skips the first player post-'l' who is better than all players pre-'l'. The winner
    is the next player who is the best among all encountered so far, including those skipped. If no such player
    is found, the player in the last position wins by default.
    """
    if not permutations:
        return np.array([])  # Return an empty array if permutations list is empty

    number_of_players = len(permutations[0])  # Determine the number of players from the length of a permutation
    wins = np.zeros(number_of_players)
    for perm in permutations:
        best_rank = number_of_players + 1
        skip_next_better = True
        for j in range(0, l):
            if perm[j] < best_rank:
                best_rank = perm[j]
        best_l = best_rank
        for k in range(l, number_of_players):
            if perm[k] < best_rank:
                if skip_next_better:
                    skip_next_better = False
                    best_rank = perm[k]
                else:
                    best_rank = perm[k]
                    wins[best_rank - 1] += 1
                    break
        if best_l == best_rank:
            wins[perm[number_of_players - 1] - 1] += 1

    win_percents = wins / len(permutations)
    return win_percents

#
# n = 10
# l = 3
# alpha = l+1
# k = 3
#
# perms = create_custom_permutations(n, k, alpha)
# emp_classic_wins = classic_stopping_rule_empirical_wins(perms, l)
# emp_modified_wins = revised_stopping_rule_empirical_wins(perms, l)
# print(emp_classic_wins)
# print("---------------------")
# print(emp_modified_wins)

# To do
# Check the success rates for different n and k

# first compare the different values of n and same k and we are comparing the wins of player 1 (so index 0 of the array)
# for the different values of n

# Initialize lists to store the results
n_values = range(5, 11)
classic_wins = []
modified_wins = []

k = 2


for n in n_values:
    l = int(np.ceil(n / np.e))
    alpha = l + 1
    perms = create_custom_permutations(n, 2, alpha)
    emp_classic_wins = classic_stopping_rule_empirical_wins(perms, l)
    emp_modified_wins = revised_stopping_rule_empirical_wins(perms, l)
    classic_wins.append(emp_classic_wins[0])
    modified_wins.append(emp_modified_wins[0])

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(n_values, classic_wins, label='Classic Wins', marker='o')
plt.plot(n_values, modified_wins, label='Modified Wins', marker='s')
plt.xlabel('n')
plt.ylabel('Empirical Wins')
plt.title('Empirical Wins vs. n')
plt.legend()
plt.grid(True)
plt.show()


