from IntermediateWork import Functions

""" New Ideas """
# For the game to represent life more realistically, you can change the game such that decision to say no is not final
# that you can have a present best and then keep searching but the more you search increases probabiltiy that you can
# not go back and get the present best so may lose out
# Can be a function of the rating of current one and number of additional searches
# How do you rate? Becomes difficult, can make it multidimesional optimisaton but then just becomes one function to give
# overall score
# There is also probability that candidate may explore other options although this is basically the probability that you
# get a know when you ask again because of passage of time
# 

# can use this present best approach or how do you approach the same game but the number of players is very high and
# there is a cost for searching, problem is the cost for searching could change things a lot, although still interesting
# to graph, this can then be used in search algorithms potentially

# look for best way to graph different asymptotics and maths functions since we have all the formulas

""" Inputs """

n = 4 # number of players
k = 2 # this the rank of the strategic candidate, with 1 being the best

# The employer will choose an l
# The candidate will choose an alpha

# We know for each candidate given an l, whether he prefers to be at alpha = 1 or alpha = n - l

# We can calculate the probability for each l and k and alpha, that of the employer picking the best candidate

# we then find the l that maximises this probability and choose it

best_probability = 0
best_l = 0

# for l in range(1, n-2):
#     alpha = 1
#     expected_payoffs = Functions.expected_payoffs(n, l, 1)
#     if expected_payoffs[k-1:0] >= expected_payoffs[k-1:0]:
#         alpha = 1
#     elif expected_payoffs[k-1:0] < expected_payoffs[k-1:0]:
#         alpha = n-l-1
#     perms = Functions.create_custom_permutations(n, k, l, alpha)
#     win_percents = Functions.empirical_wins(perms, n, l)
#     if win_percents[0] > best_probability:
#         best_probability = win_percents[0]
#         best_l = l
#
#
# print(best_probability)
# print(best_l)

# Check if this works by comparing to it done by hand

# if it works, we can then make this a function and loop through all the different k's and find the l that performs
# the best given a uniform distribution over all possible k's

n = 10

for k in range(1,n-1):
    for l in range(1, n - 2):
        alpha = 1
        expected_payoffs = Functions.expected_payoffs(n, l, 1)
        if expected_payoffs[k - 1:0] >= expected_payoffs[k - 1:0]:
            alpha = 1
        elif expected_payoffs[k - 1:0] < expected_payoffs[k - 1:0]:
            alpha = n - l - 1
        perms = Functions.create_custom_permutations(n, k, l, alpha)
        win_percents = Functions.empirical_wins(perms, n, l)
        if win_percents[0] > best_probability:
            best_probability = win_percents[0]
            best_l = l

print(best_probability)
print(best_l)