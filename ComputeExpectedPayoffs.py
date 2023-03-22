import math
import numpy as np
# import matplotlib.pyplot
import pandas as pd

e = math.e

n = 6
l = 5 # int(round(n/e, 0)) # 2
print(l)
print(int(round(n/e, 0)))

payoffs1 = np.empty((n, 2))
payoffs = np.empty((n, 2))

for k in range(1, n+1):
    if (n - k - l > 0):
        # print(n)
        # print(k)
        # print(l)
        x = (math.factorial(n-1-l)*math.factorial(n-k))/(math.factorial(n-k-l)*math.factorial(n-1))
        # print(math.factorial(n-1-l)*math.factorial(n-k))
        # print(math.factorial(n-k-l)*math.factorial(n-1))
        # print((math.factorial(n-1-l)*math.factorial(n-k))/(math.factorial(n-k-l)*math.factorial(n-1)))
        payoffs1[k - 1, 0] = x
        payoffs1[k - 1 , 1] = (l/(n-1))
    else:
        # print(n)
        # print(k)
        # print(l)
        payoffs1[k - 1, 0] = 0
        payoffs1[k - 1, 1] = (l/(n-1))

payoffs1[(n-l):n, 0] = 0

# print(np.round(payoffs1, decimals=3))
# print("Now including alpha")


# print(list(range(n)))

# payoffs[0:n, 0] = (1-(l/n))**(np.array(list(range(n))))
# payoffs[0:n, 0] = (math.factorial(n-1-l)*math.factorial(n-k))/(math.factorial(n-k-l))
# payoffs[0:n, 1] = (l/(n-1))
# payoffs[(n-l):n, 0] = 0

# print(payoffs1)
# print(np.round(payoffs, decimals=5))



def prob_best(x):
    return -x*np.log(x)

# print(prob_best(.5))
""" Pass an array that goes from 0 to 1 to prob_best function and plot"""
# 1/e maximises the function

# Write Program to compute best l for given sample size n


def expected_payoffs(n, l, alpha):
    for k in range(1, n + 1):
        if (n - k - l - alpha + 1 > 0):
            # print(n)
            # print(k)
            # print(l)
            upper = (math.factorial(l) * math.factorial(alpha - 1) * math.factorial(n - l - alpha) * math.factorial(n - k))
            lower = (math.factorial(n - 1) * math.factorial(n - k - l - alpha + 1) * math.factorial(l + alpha - 1))
            # print(math.factorial(n-1-l)*math.factorial(n-k))
            # print(math.factorial(n-k-l)*math.factorial(n-1))
            # print((math.factorial(n-1-l)*math.factorial(n-k))/(math.factorial(n-k-l)*math.factorial(n-1)))
            # print(upper)
            # print(lower)
            payoffs[k - 1, 0] = upper/lower
            payoffs[k - 1, 1] = (l / (n - 1))
        else:
            # print(n)
            # print(k)
            # print(l)
            payoffs[k - 1, 0] = 0
            payoffs[k - 1, 1] = (l / (n - 1))

    payoffs[(n - l):n, 0] = 0

    return np.round(payoffs, decimals=3)


payoffs = expected_payoffs(n, l, alpha=1)

# print(expected_payoffs(n, l, alpha=1))

print(payoffs)

index = []
for a in range(0, n):
    current_l = str(a+1)
    index.append("n = " + current_l)
print(index)

# columns = []
# for a in range(0, n):
#     current_l = str(a+1)
#     columns.append("l = " + current_l)
#
# print(columns)

df = pd.DataFrame(payoffs, columns=['alpha = 1', 'alpha = 5'], index=index )
# print(df)
print(df.to_latex(index=True))

""" Finish this Code later, for now it is easy to do manually"""
# preferred = []
# for b in range(0,n):
#     position = str(n-(b+1))
#     payoffs = expected_payoffs(n, b, 1)
#     for a in range(0, n):
#         if payoffs[a][0] >= payoffs[a][1]:
#             preferred.append("alpha = 1")
#         else:
#             preferred.append("alpha = " + position)
#
# print(preferred)


# print(payoffs[1])
# print("here is the second")
# print(expected_payoffs(n, l , alpha=2))


def amount_preferring_alpha1(payoffs):
    amount = 0
    for i in range(0, len(payoffs)):
        if payoffs[i, 0] > payoffs[i, 1]:
            amount += 1
    return amount


print(amount_preferring_alpha1(payoffs))