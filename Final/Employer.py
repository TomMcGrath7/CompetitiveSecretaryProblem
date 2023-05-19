import math
import numpy as np
import pandas as pd

""" Probability of employer picking the best candidate given alpha = 1 is chosen"""
" This function takes as input "
" number of players n "
" number of candidates to be rejected l"
" The rank of the strategic candidate k"

" This function is split into 3 cases, when k = 1, when k is between 2 and n-l+1 and when k is greater than n-l+1."
" In the first case the probability of choosing the best is 1"
"In the second case the probability of choosing the best is summing all the times the best does not appear in the " \
"first l and comes before all those better than k "
"In the third case the probability of choosing the best is like the regular secretary problem as this candidate k has " \
"no chance of "


def probability_best_chosen(n, l, k):
    if k == 1:
        return 1
    if l == 0 and k != 1:
        return 0
    elif (1 < k < (n - l + 1)):  # confirm the right bound
        numerator = 0
        for i in range(2, k):
            numerator += (1 / (i - 1)) * ((math.factorial(l) * math.factorial(n - l - 1) * math.factorial(n - i - 1)) /
                                          (math.factorial(l - 1) * math.factorial(n - l - i)))
        denominator = math.factorial(n - 1)
        return numerator / denominator
    elif k == n-l+1:
        numerator = 0
        for i in range(2, k):
            numerator += (1 / (i - 1)) * ((math.factorial(l) * math.factorial(n - l - 1) * math.factorial(n - i - 1)) /
                                          (math.factorial(l - 1) * math.factorial(n - l - i)))
        denominator = math.factorial(n - 1)
        a = numerator / denominator
        sum = 0
        for i in range(l + 1, n):
            sum += (1 / (i - 1))
        b = (l / (n - 1)) * sum
        return max(a, b)
    elif (k > (n - l + 1)):
        sum = 0
        for i in range(l + 1, n):
            sum += (1 / (i - 1))
        return (l / (n - 1)) * sum


n = 10
l = 1
alpha = 1
k = 1

print(probability_best_chosen(n, l, k))

probabilities = np.zeros(n)

for i in range(0,n):
    probabilities[i] = probability_best_chosen(n,l,i+1)
print(probabilities)


def tableFi(probability_best):
    index = []
    for a in range(0, n):
        current_k = str(a + 1)
        index.append("k = " + current_k)

    df = pd.DataFrame(probability_best, columns=['Probability picking best'], index=index)
    return df


df = tableFi(probabilities)
print(df)
print(df.to_latex(index=True))

