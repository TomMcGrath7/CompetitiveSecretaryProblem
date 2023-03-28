import numpy as np
import itertools
import Functions
import math
import pandas as pd


""" Inputs """
n = 6
l = 1
k = 3
alpha = 1

# candidates = np.array(list(range(1, n+1)))
# print(candidates)
#
# # Fix position of player k
# fixed_player = candidates[k-1]
# print(fixed_player)
# print(k-1)
#
# # Remove fixed player from candidates
# candidates = np.delete(candidates, k-1)
#
# # Generate all permutations of candidates without fixed player
# perms = list(itertools.permutations(candidates))
# print(perms)
#
# # Add fixed player back to each permutation
# # perms = [np.insert(perm, l+alpha-1, fixed_player) for perm in perms]
# # perms = [(fixed_player,) + perm for perm in perms]
# perms1 = [perm[:l+alpha-1] + (fixed_player,) + perm[l+alpha-1:] for perm in perms]
# print(perms1)
#
# print("break")


def create_custom_permutations(number_of_players, k, l, alpha):
    candidates = np.array(list(range(1, number_of_players + 1)))
    # print(candidates)
    fixed_player = candidates[k-1]
    # print(fixed_player)
    candidates = np.delete(candidates, k - 1)
    perms = list(itertools.permutations(candidates))
    # print(perms)
    perms = [perm[:l + alpha - 1] + (fixed_player,) + perm[l + alpha - 1:] for perm in perms]
    return perms


perms = create_custom_permutations(n, k, l, alpha)
# print(perms)
# Compute probability that item 1 is picked each time
# Count how many times item is the winner

# wins = np.zeros(n)
# # print(wins)
#
# for i in range(len(perms)):
#     best_rank = n
#     for j in range(0, l):
#         if perms[i][j] < best_rank:
#             best_rank = perms[i][j]
#     best_l = best_rank
#     for k in range(l, n):
#         if perms[i][k] < best_rank:
#             best_rank = perms[i][k]
#             wins[best_rank-1] += 1
#             break
#     if best_l == best_rank:
#         wins[perms[i][n-1]-1] += 1
#
# win_percents = wins/len(perms)
# print(win_percents)
#
# print("Break")

# put all this in a function
# then call the function for different l's and for the different l look at when probability of beiing picked for the k puts him at the end
# and it results in a higher probability of picking the best one


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


def proability_picking_best(n, l, k):
    sum = 0
    for i in range(2, k-2):
        print(i)
        sum += (math.factorial(l)*math.factorial(n-l-1)*(math.factorial(n-(i+1))/(math.factorial(l-1)*math.factorial(n-i-l))))*(1/(i-1))
    denominator = math.factorial(n-1)
    return sum/denominator


# prob_winning = proability_picking_best(n, l, k)
# print(prob_winning)

n = 6
l = 3
k = 2
alpha = 3

win_percents = empirical_wins(perms, n, l)
print(win_percents[0])

percentage = []
for a in range(1, n+1): # for each k
    row = []
    for b in range(1, n): # for each l
        alpha = n-l
        perms = create_custom_permutations(n, a, b, alpha)
        pick_percent = empirical_wins(perms, n, b)[0]
        row.append(pick_percent)
    percentage.append(row)


print(percentage)

index = []
for a in range(0, n):
    current_l = str(a+1)
    index.append("n = " + current_l)
print(index)

columns = []
for a in range(0, n-1):
    current_l = str(a+1)
    columns.append("l = " + current_l)

print(columns)

df = pd.DataFrame(percentage, columns=columns, index=index )
print(df)
# print(df.to_latex(index=True))


def classic_probability_picking_best(n, l):
    sum = 0
    for a in range(l, n):
        sum += (1/(a-1))
    return (((l)-1)/n)*sum


# print(classic_probability_picking_best(100000000, round(100000000/math.e)))
# print(1/math.e)
# print(classic_probability_picking_best(100000000, round(100000000/math.e)) - 1/math.e)

print(classic_probability_picking_best(5, 2))