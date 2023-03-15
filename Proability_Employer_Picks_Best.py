import numpy as np
import itertools

""" Inputs """
n = 4
l = 1
k = 4
alpha = 1

candidates = np.array(list(range(1, n+1)))
print(candidates)

# Fix position of player k
fixed_player = candidates[k-1]

# Remove fixed player from candidates
candidates = np.delete(candidates, k-1)

# Generate all permutations of candidates without fixed player
perms = list(itertools.permutations(candidates))
print(perms)

# Add fixed player back to each permutation
# perms = [np.insert(perm, l+alpha-1, fixed_player) for perm in perms]
# perms = [(fixed_player,) + perm for perm in perms]
perms = [perm[:l+alpha-1] + (fixed_player,) + perm[l+alpha-1:] for perm in perms]
print(perms)

# Compute probability that item 1 is picked each time
# Count how many times item is the winner

wins = np.zeros(n)
print(wins)

for i in range(len(perms)):
    best_rank = n
    for j in range(0, l):
        if perms[i][j] < best_rank:
            best_rank = perms[i][j]
    best_l = best_rank
    for k in range(l, n):
        if perms[i][k] < best_rank:
            best_rank = perms[i][k]
            wins[best_rank-1] += 1
            break
    if best_l == best_rank:
        wins[perms[i][n-1]-1] += 1

win_percents = wins/len(perms)
print(win_percents)

# put all this in a function
# then call the function for different l's and for the different l look at when probability of beiing picked for the k puts him at the end
# and it results in a higher probability of picking the best one