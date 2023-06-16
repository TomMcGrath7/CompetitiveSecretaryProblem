import math
import numpy as np
import matplotlib.pyplot as plt


def payoff(n, l, alpha, k):
    if n == 1:
        return 1
    if alpha == 1 and k < (n - l + 1):
        return (math.factorial(l) * math.factorial(n - l - 1) * math.factorial(n - k)) / (
                    math.factorial(n - 1) * math.factorial(l) * math.factorial(n - k - l))
    elif alpha == 1 and k >= (n - l + 1):
        return 0
    elif alpha == (n - l):
        return l / (n - 1)


# test = payoff(10, 3, 7, 2)
# print(test)


def uniform_weight(cumulative_probability):
    return cumulative_probability / len(cumulative_probability)


def robust_alpha(n_max):
    output = np.zeros((n_max + 1, 3))
    for n in range(1, n_max + 1):
        l = round(n / math.e)
        print("l is")
        print(l)
        unweighted = np.ones(n)
        weights = uniform_weight(unweighted)
        print("Weights are")
        print(weights)
        alpha_1 = 0
        alpha_n_l = 0
        for k in range(1, n + 1):
            print("k is")
            print(k)
            alpha_1 += payoff(n, l, 1, k) * weights[k-1]
            alpha_n_l += payoff(n, l, n - l, k) * weights[k-1]
            print("Payoff for alpha = 1 is")
            print(payoff(n, l, 1, k))
            print("Payoff for alpha = n - l is")
            print(payoff(n, l, n - l, k))

        output[n, 0] = n
        if alpha_1 > alpha_n_l:
            output[n, 1] = 0 # 0 means alpha_1 is the robust alpha
            output[n, 2] = alpha_1
        elif alpha_1 <= alpha_n_l:
            output[n, 1] = 1 # 1 means alpha_n_l is the robust alpha
            output[n, 2] = alpha_n_l
        print("output is")
        print(output)
    return output



e = math.e
# print(e)
n_max = 100 
# l = round(n_max/e)
# print(l)

output = robust_alpha(n_max)
output = output[1:]
print(output)

# PLOT THE RESULTS
# Create the plot
plt.scatter(output[0:, 0], output[0:, 2], c='blue', label='Selected Probability')
# plt.scatter(output[:, 0], output[:, 1], c='red', label='Robust Alpha')
plt.xlabel('Number of Candidates')
plt.ylabel('Selection Probability')
plt.title('Robust Alpha for different Number of Candidates')
plt.legend()
plt.ylim(-0.03, 1.03)

# Display the plot
plt.show()




