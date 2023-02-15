import numpy as np
import itertools

""" Inputs """
n = 4
l = 2

candidates = np.array(list(range(1, n+1)))
print(candidates)

perms = list(itertools.permutations(candidates))

print(perms)

wins = np.zeros(n)
print(wins)

for i in range(len(perms)):
    best_rank = n
    for j in range(0, l):
        if perms[i][j] < best_rank:
            best_rank = perms[i][j]
    for k in range(l, n):
        if perms[i][k] < best_rank:
            wins[best_rank-1] += 1
            break
        else:
            wins[perms[i][k]-1] += 1 # Unsure about this code, check logic later


win_percents = wins/len(perms)

# Not working
print(win_percents)

