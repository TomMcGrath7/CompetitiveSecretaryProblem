import numpy as np
import math
import itertools
import pandas as pd
import matplotlib.pyplot as plt


def probability_best_chosen_BR(n, l, k):
    if k == 1:
        return 1
    if l == 0 and k != 1:
        return 0
    elif 1 < k < (n - l + 1):
        # print(n)
        # print(l)
        # print(k)
        if l < (math.factorial(n - l - 1) * math.factorial(n - k)) / (
                math.factorial(n - 2) * math.factorial(n - k - l)):  # confirm the right bound
            numerator = 0
            for i in range(2, k):
                numerator += (1 / (i - 1)) * (
                        (math.factorial(l) * math.factorial(n - l - 1) * math.factorial(n - i - 1)) /
                        (math.factorial(l - 1) * math.factorial(n - l - i)))
            denominator = math.factorial(n - 1)
            return numerator / denominator
        elif l == (math.factorial(n - l - 1) * math.factorial(n - k)) / (
                math.factorial(n - 2) * math.factorial(n - k - l)):
            numerator = 0
            for i in range(2, k):
                numerator += (1 / (i - 1)) * (
                        (math.factorial(l) * math.factorial(n - l - 1) * math.factorial(n - i - 1)) /
                        (math.factorial(l - 1) * math.factorial(n - l - i)))
            denominator = math.factorial(n - 1)
            a = numerator / denominator
            sum = 0
            for i in range(l + 1, n):
                sum += (1 / (i - 1))
            b = (l / (n - 1)) * sum
            return max(a, b)
        elif l >= (math.factorial(n - l - 1) * math.factorial(n - k)) / (
                math.factorial(n - 2) * math.factorial(n - k - l)):
            sum = 0
            for i in range(l + 1, n):
                sum += (1 / (i - 1))
            return (l / (n - 1)) * sum
    elif k >= (n - l + 1):
        sum = 0
        for i in range(l + 1, n):
            sum += (1 / (i - 1))
        return (l / (n - 1)) * sum
    else:
        print("Error")
        return "Error"


def probability_best_chosen_BR_BIG(n, l, k):
    if k == 1:
        return 1
    if l == 0 and k != 1:
        return 0
    elif 1 < k < (n - l + 1):
        if l < math.exp(math.lgamma(n - l - 1) + math.lgamma(n - k) - math.lgamma(n - 2) - math.lgamma(n - k - l)):
            numerator = 0
            for i in range(2, k):
                numerator += (1 / (i - 1)) * (
                    math.exp(math.lgamma(l) + math.lgamma(n - l - 1) + math.lgamma(n - i - 1) - math.lgamma(
                        l - 1) - math.lgamma(n - l - i)))
            denominator = math.factorial(n - 1)
            return numerator / denominator
        elif l == math.exp(math.lgamma(n - l - 1) + math.lgamma(n - k) - math.lgamma(n - 2) - math.lgamma(n - k - l)):
            numerator = 0
            for i in range(2, k):
                numerator += (1 / (i - 1)) * (
                    math.exp(math.lgamma(l) + math.lgamma(n - l - 1) + math.lgamma(n - i - 1) - math.lgamma(
                        l - 1) - math.lgamma(n - l - i)))
            denominator = math.factorial(n - 1)
            a = numerator / denominator
            s = 0
            for i in range(l + 1, n):
                s += (1 / (i - 1))
            b = (l / (n - 1)) * s
            return max(a, b)
        elif l >= math.exp(math.lgamma(n - l - 1) + math.lgamma(n - k) - math.lgamma(n - 2) - math.lgamma(n - k - l)):
            s = 0
            for i in range(l + 1, n):
                s += (1 / (i - 1))
            return (l / (n - 1)) * s
    elif k >= (n - l + 1):
        s = 0
        for i in range(l + 1, n):
            s += (1 / (i - 1))
        return (l / (n - 1)) * s
    else:
        print("Error")
        return "Error"


def probability_best_chosen_BR_BIG2(n, l, k):
    if k == 1:
        return 1
    if l == 0 and k != 1:
        return 0
    elif 1 < k < (n - l + 1):
        if l < (math.comb(n - l - 1, n - k) / math.comb(n - 2, n - k - l)):
            numerator = 0
            for i in range(2, k):
                numerator += (1 / (i - 1)) * (
                    (math.comb(l, l - 1) * math.comb(n - l - 1, n - l - i) * math.comb(n, n - i - 1)))
            denominator = math.factorial(n - 1)
            return numerator / denominator
        elif l == (math.comb(n - l - 1, n - k) / math.comb(n - 2, n - k - l)):
            numerator = 0
            for i in range(2, k):
                numerator += (1 / (i - 1)) * (
                    (math.comb(l, l - 1) * math.comb(n - l - 1, n - l - i) * math.comb(n, n - i - 1)))
            denominator = math.factorial(n - 1)
            a = numerator / denominator
            sum = 0
            for i in range(l + 1, n):
                sum += (1 / (i - 1))
            b = (l / (n - 1)) * sum
            return max(a, b)
        elif l >= (math.comb(n - l - 1, n - k) / math.comb(n - 2, n - k - l)):
            sum = 0
            for i in range(l + 1, n):
                sum += (1 / (i - 1))
            return (l / (n - 1)) * sum
    elif k >= (n - l + 1):
        sum = 0
        for i in range(l + 1, n):
            sum += (1 / (i - 1))
        return (l / (n - 1)) * sum
    else:
        print("Error")
        return "Error"


from decimal import Decimal, getcontext

# Set precision.
getcontext().prec = 100


def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


import logging
import cProfile
import pstats
from math import factorial

# Set up logging
logging.basicConfig(filename='debug.log', level=logging.DEBUG)


def probability_best_chosen_BR_BIG3(n, l, k):
    n = Decimal(n)
    l = Decimal(l)
    k = Decimal(k)
    if k == 1:
        return Decimal(1)
    if l == 0 and k != 1:
        return Decimal(0)
    elif 1 < k < (n - l + 1):
        if l < (factorial(n - l - 1) * factorial(n - k)) / (
                factorial(n - 2) * factorial(n - k - l)):
            numerator = Decimal(0)
            for i in range(2, int(k)):
                numerator += (Decimal(1) / (Decimal(i) - Decimal(1))) * (
                        (factorial(l) * factorial(n - l - Decimal(1)) * factorial(n - Decimal(i) - Decimal(1))) /
                        (factorial(l - Decimal(1)) * factorial(n - l - Decimal(i))))
            denominator = factorial(n - Decimal(1))
            return numerator / denominator
        elif l == (factorial(n - l - 1) * factorial(n - k)) / (
                factorial(n - 2) * factorial(n - k - l)):
            numerator = Decimal(0)
            for i in range(2, int(k)):
                numerator += (Decimal(1) / (Decimal(i) - Decimal(1))) * (
                        (factorial(l) * factorial(n - l - Decimal(1)) * factorial(n - Decimal(i) - Decimal(1))) /
                        (factorial(l - Decimal(1)) * factorial(n - l - Decimal(i))))
            denominator = factorial(n - Decimal(1))
            a = numerator / denominator
            sum = Decimal(0)
            for i in range(int(l) + 1, int(n)):
                sum += (Decimal(1) / (Decimal(i) - Decimal(1)))
            b = (l / (n - Decimal(1))) * sum
            return max(a, b)
        elif l >= (factorial(n - l - 1) * factorial(n - k)) / (
                factorial(n - 2) * factorial(n - k - l)):
            sum = Decimal(0)
            for i in range(int(l) + 1, int(n)):
                sum += (Decimal(1) / (Decimal(i) - Decimal(1)))
            return (l / (n - Decimal(1))) * sum
    elif k >= (n - l + 1):
        sum = Decimal(0)
        for i in range(int(l) + 1, int(n)):
            sum += (Decimal(1) / (Decimal(i) - Decimal(1)))
        return (l / (n - Decimal(1))) * sum
    else:
        print("Error")
        return "Error"


def probability_best_chosen_BR_BIG4(n, l, k):
    n = np.float64(n)
    l = np.float64(l)
    k = np.float64(k)
    if k == 1:
        return np.float64(1)
    if l == 0 and k != 1:
        return np.float64(0)
    elif 1 < k < (n - l + 1):
        if l < (np.prod(np.arange(n - l, n - l - 1, -1)) * np.prod(np.arange(n - k, n - k, -1))) / (
                np.prod(np.arange(n - 1, n - 2, -1)) * np.prod(np.arange(n - k - l, n - k - l, -1))):
            numerator = np.float64(0)
            for i in np.arange(2, k):
                numerator += (np.float64(1) / (i - np.float64(1))) * (
                        (np.prod(np.arange(l, l - 1, -1)) * np.prod(np.arange(n - l - 1, n - l - i, -1)) * np.prod(
                            np.arange(n - i - 1, n - i - 1, -1))) /
                        (np.prod(np.arange(l - 1, l - 1, -1)) * np.prod(np.arange(n - l - i, n - l - i, -1))))
            denominator = np.prod(np.arange(n - 1, n - 1, -1))
            return numerator / denominator
        elif l == (np.prod(np.arange(n - l - 1, n - l - 1, -1)) * np.prod(np.arange(n - k, n - k, -1))) / (
                np.prod(np.arange(n - 2, n - 2, -1)) * np.prod(np.arange(n - k - l, n - k - l, -1))):
            numerator = np.float64(0)
            for i in np.arange(2, k):
                numerator += (np.float64(1) / (i - np.float64(1))) * (
                        (np.prod(np.arange(l, l - 1, -1)) * np.prod(np.arange(n - l - 1, n - l - i, -1)) * np.prod(
                            np.arange(n - i - 1, n - i - 1, -1))) /
                        (np.prod(np.arange(l - 1, l - 1, -1)) * np.prod(np.arange(n - l - i, n - l - i, -1))))
            denominator = np.prod(np.arange(n - 1, n - 1, -1))
            a = numerator / denominator
            sum = np.float64(0)
            for i in np.arange(l + 1, n):
                sum += (np.float64(1) / (i - np.float64(1)))
            b = (l / (n - np.float64(1))) * sum
            return max(a, b)
        elif l >= (np.prod(np.arange(n - l - 1, n - l - 1, -1)) * np.prod(np.arange(n - k, n - k, -1))) / (
                np.prod(np.arange(n - 2, n - 2, -1)) * np.prod(np.arange(n - k - l, n - k - l, -1))):
            sum = np.float64(0)
            for i in np.arange(l + 1, n):
                sum += (np.float64(1) / (i - np.float64(1)))
            return (l / (n - np.float64(1))) * sum
    elif k >= (n - l + 1):
        sum = np.float64(0)
        for i in np.arange(l + 1, n):
            sum += (np.float64(1) / (i - np.float64(1)))
        return (l / (n - np.float64(1))) * sum
    else:
        print("Error")
        return "Error"


def probability_best_chosen_BR_BIG5(n, l, k):
    n = int(n)
    l = int(l)
    k = int(k)

    # Add +1 where necessary because lgamma(n) == factorial(n-1)

    if k == 1:
        return 1
    if l == 0 and k != 1:
        return 0
    elif 1 < k < (n - l + 1):
        if l < (math.lgamma(n - l + 2) + math.lgamma(n - k + 2) - math.lgamma(n) - math.lgamma(n - k - l + 2)):
            numerators = []
            for i in range(2, k):
                numerators.append(math.log(1 / (i - 1)) +
                                  math.lgamma(l + 2) + math.lgamma(n - l + 1) + math.lgamma(n - i + 2) -
                                  (math.lgamma(l + 1) + math.lgamma(n - l - i + 2)))
            if numerators:
                a = max(numerators)
                numerator = a + math.log(sum(math.exp(x - a) for x in numerators))
            else:
                numerator = 0
            denominator = math.lgamma(n + 1)
            return math.exp(numerator - denominator)
        elif l == (math.lgamma(n - l + 2) + math.lgamma(n - k + 2) - math.lgamma(n) - math.lgamma(n - k - l + 2)):
            numerators = []
            for i in range(2, k):
                numerators.append(math.log(1 / (i - 1)) +
                                  math.lgamma(l + 2) + math.lgamma(n - l + 1) + math.lgamma(n - i + 2) -
                                  (math.lgamma(l + 1) + math.lgamma(n - l - i + 2)))
            if numerators:
                a = max(numerators)
                numerator = a + math.log(sum(math.exp(x - a) for x in numerators))
            else:
                numerator = 0
            denominator = math.lgamma(n + 1)
            a = math.exp(numerator - denominator)
            summ = 0
            for i in range(l + 1, n):
                summ += (1 / (i - 1))
            b = (l / (n - 1)) * summ
            return max(a, b)
        elif l >= (math.lgamma(n - l + 2) + math.lgamma(n - k + 2) - math.lgamma(n) - math.lgamma(n - k - l + 2)):
            summ = 0
            for i in range(l + 1, n):
                summ += (1 / (i - 1))
            return (l / (n - 1)) * summ
    elif k >= (n - l + 1):
        summ = 0
        for i in range(l + 1, n):
            summ += (1 / (i - 1))
        return (l / (n - 1)) * summ
    else:
        print("Error")
        return "Error"


""" This function finds the best l for a given number of applicants"""
"It goes through each possible l and calculates the sum of probabilities for that l and n for each k"


def best_l(n):
    l_probs = np.zeros(n)
    unweighted = np.ones(n)
    weights = uniform_weight(unweighted)
    if n == 1:
        max_l = 0
    elif n % 2 == 0:
        max_l = int(n / 2)
    elif n % 2 != 0:
        max_l = int((n - 1) / 2)
    for l in range(0, (max_l + 1)):
        summ = 0
        for k in range(1, n + 1):
            # summ += probability_best_chosen_BR(n, l, k) * weights[k-1]
            summ += probability_best_chosen_BR_BIG3(n, l, k) * Decimal(weights[k - 1])
        l_probs[l] = summ
    return l_probs


" This is for printing the best responses and employer utility"

" Here will be the plot for fixed n and vary l. should be quadratic"
" so we get an array with employer utility and l"


def varied_l(n):
    columns = 2
    output = np.zeros((n, columns))
    for l in range(0, n):
        summ = 0
        for k in range(1, n + 1):
            summ += probability_best_chosen_BR(n, l, k)
            # Get prob. of picking best
            # for every l get the probability of picking the best
        output[l, 0] = l / n
        output[l, 1] = summ / n
    return output


def many_n(n_max):
    output = np.zeros((n_max + 1, 3))
    for n in range(1, n_max + 1):
        print(n)
        l_probs = best_l(n)
        max_value = np.max(l_probs)
        best_l_2 = np.argmax(l_probs)
        output[n, 0] = n
        output[n, 1] = best_l_2 / n
        output[n, 2] = max_value

    return output


def uniform_weight(cumulative_probability):
    return cumulative_probability / len(cumulative_probability)


n_test = 250
output = many_n(n_test)
output = output[1:]
print(output)

weights = np.ones(n_test)
weights = uniform_weight(weights)

# Create the plot
plt.scatter(output[:, 0], output[:, 1], c='blue', label='Search Fraction')
plt.scatter(output[:, 0], output[:, 2], c='red', label='Probability Picking best')
plt.axhline(1 / np.e, color='gray', linestyle='dotted', alpha=0.3, label='1/e')
plt.xlabel('Number of Candidates')
plt.ylabel('Probability/Search Fraction')
plt.title('Decaying Decreasing Secretary Problem')
plt.legend()
plt.ylim(-0.03, 1.03)  # Set the y-axis limits to 0 and 1

# Add the faint line for 'weights' across the x-axis
plt.plot(output[0:, 0], weights, color='gray', linewidth=0.5, alpha=0.5)
# Display the plot
plt.show()

print(1 / math.e)
