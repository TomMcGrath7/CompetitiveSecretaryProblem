import numpy as np
import itertools

""" Inputs """
n = 10
l = 4

candidates = np.array(list(range(1, n+1)))
print(candidates)

perms = list(itertools.permutations(candidates))

# print(perms)

wins = np.zeros(n)
print(wins)

for i in range(len(perms)):
    best_rank = n
    for j in range(0, l):
        # print("j is " + str(j))
        if perms[i][j] < best_rank:
            # print("j in the if is " + str(j))
            # print("Old best rank " + str(best_rank))
            best_rank = perms[i][j]
            # print("New best rank " + str(best_rank))
    best_l = best_rank
    for k in range(l, n):
        # print("K is " + str(k))
        if perms[i][k] < best_rank:
            best_rank = perms[i][k]
            # print("k in the if is " + str(k))
            # print("winner is " + str(perms[i][k]))
            wins[best_rank-1] += 1
            break
    if best_l == best_rank:
        # print("There was no winnner, so the person at the end is " + str(perms[i][n-1]))
        wins[perms[i][n-1]-1] += 1 # Unsure about this code, check logic later
    # print("the current count of wins is ")
    # print(wins)

win_percents = wins/len(perms)

# Not working
print(win_percents)

""" To DO """

# Plot graphs of the different probabilities as the different parameters change

# Lastly, can we find an l that is robust to any k, so the choice of k is unknown to the employer
