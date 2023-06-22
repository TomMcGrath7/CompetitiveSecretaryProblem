import numpy as np
import math
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import cProfile, pstats, io
import logging
import os
# set up logging
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

# All your function definitions go here
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


from decimal import Decimal, getcontext

# Set precision.
getcontext().prec = 100


# def factorial(n):
#     if n == 0:
#         return 1
#     else:
#         return n * factorial(n - 1)

import functools


@functools.lru_cache(maxsize=None)
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


def probability_best_chosen_BR_BIG3(n, l, k):
    # n = Decimal(n)
    # l = Decimal(l)
    # k = Decimal(k)
    """ Try these lines if you want to try make it even faster"""
    # factorials = [1, 1]
    # for i in range(2, int(n) + 1):
    #     factorials.append(factorials[-1] * i)
    """Now, you can use factorials[i] instead of factorial(i) in your function."""
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
            factorial_n_minus_l_minus_1 = factorial(n - l - Decimal(1))
            factorial_l = factorial(l)
            for i in range(2, int(k)):
                # numerator += (Decimal(1) / (Decimal(i) - Decimal(1))) * (
                #         (factorial(l) * factorial(n - l - Decimal(1)) * factorial(n - Decimal(i) - Decimal(1))) /
                #         (factorial(l - Decimal(1)) * factorial(n - l - Decimal(i))))
                factorial_n_minus_i_minus_1 = factorial(n - Decimal(i) - Decimal(1))
                # Use precalculated factorials inside the loop
                numerator += (Decimal(1) / (Decimal(i) - Decimal(1))) * (
                        (factorial_l * factorial_n_minus_l_minus_1 * factorial_n_minus_i_minus_1) /
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
                sum += (Decimal(1) / (i - Decimal(1)))
            return (l / (n - Decimal(1))) * sum
    elif k >= (n - l + 1):
        sum = Decimal(0)
        for i in range(int(l) + 1, int(n)):
            sum += 1 / (i - Decimal(1))
        return (l / (n - Decimal(1))) * sum
    else:
        print("Error")
        return "Error"


def best_l(n):
    memo = {}
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
            key = (n, l, k)
            if key not in memo:
                memo[key] = probability_best_chosen_BR_BIG3(n, l, k)
            summ += memo[key] * Decimal(weights[k - 1])
            # summ += probability_best_chosen_BR_BIG3(n, l, k) * Decimal(weights[k - 1])
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for n in range(1, n_max + 1):
        # print(n)
        l_probs = best_l(n)
        max_value = np.max(l_probs)
        best_l_2 = np.argmax(l_probs)

        output[n, 0] = n
        output[n, 1] = best_l_2 / n
        output[n, 2] = max_value
        print(output[n, :])
        if n % 10 == 0:  # Every 10 iterations
            np.save(os.path.join(script_dir, 'output.npy'), output)

    return output


def uniform_weight(cumulative_probability):
    return cumulative_probability / len(cumulative_probability)

# Then start the main execution of your program in the main() function
def main():
    np.load('output.npy')
    pr = cProfile.Profile()
    pr.enable()

    # Code execution starts here
    n_test = 250
    output = many_n(n_test)
    output = output[1:]
    print(output)

    weights = np.ones(n_test)
    weights = uniform_weight(weights)

    # Create the plot
    plt.scatter(output[:, 0], output[:, 1], c='blue', s=0.5, label='Search Fraction')
    plt.scatter(output[:, 0], output[:, 2], c='red', s=0.5, label='Probability Picking best')
    plt.axhline(1 / np.e, color='gray', linestyle='dotted', alpha=0.3, label='1/e')
    plt.xlabel('Number of Candidates')
    plt.ylabel('Probability/Search Fraction')
    plt.title('Long Run Relationship Between l* and n')
    plt.legend()
    plt.ylim(-0.01, 1.01)  # Set the y-axis limits to 0 and 1

    # Add the faint line for 'weights' across the x-axis
    # plt.plot(output[0:, 0], weights, color='gray', linewidth=0.5, alpha=0.5)
    # Display the plot
    plt.show()

    print(1 / math.e)

    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()

    # print profiling info to the log file
    logging.info(s.getvalue())

# Run the main function
if __name__ == "__main__":
    main()