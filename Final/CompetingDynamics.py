import numpy as np
import math
import itertools
import pandas as pd
import matplotlib.pyplot as plt


#
# def probability_best_chosen_BR(n, l, k):
#     if k == 1:
#         return 1
#     if l == 0 and k != 1:
#         return 0
#     elif 1 < k <= (n - l + 1):
#         if l <= (math.factorial(n - l - 1) * math.factorial(n - k))/(math.factorial(n - 2) * math.factorial(n - k - l)):  # confirm the right bound
#             numerator = 0
#             for i in range(2, k):
#              numerator += (1 / (i - 1)) * ((math.factorial(l) * math.factorial(n - l - 1) * math.factorial(n - i - 1)) /
#                                           (math.factorial(l - 1) * math.factorial(n - l - i)))
#             denominator = math.factorial(n - 1)
#             return numerator / denominator
#         elif l > (math.factorial(n - l - 1) * math.factorial(n - k))/(math.factorial(n - 2) * math.factorial(n - k - l)):
#             sum = 0
#             for i in range(l + 1, n):
#                 sum += (1 / (i - 1))
#             return (l / (n - 1)) * sum
#     elif  k > (n - l + 1):
#         sum = 0
#         for i in range(l + 1, n):
#             sum += (1 / (i - 1))
#         return (l / (n - 1)) * sum
#     else:
#         print("Error")
#         return "Error"


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


""" This function finds the best l for a given number of applicants"""
"It goes through each possible l and calculates the sum of probabilities for that l and n for each k"


def best_l(n):
    l_probs = np.zeros(n)
    unweighted = np.ones(n)
    # weights = uniform_weight(unweighted)
    # weights = linear_decreasing_weight(unweighted)
    # weights = exponential_decreasing_weight(unweighted)
    # weights = geometric_decreasing_weight(unweighted, 0.8)
    # weights = harmonic_decreasing_weight(unweighted)
    weights = decaying_decreasing_weight(unweighted, 0.5)

    print("The weights are")
    print(weights)
    # weights = decaying_decreasing_weight(unweighted, 0.5)
    print("we are in best l")
    print(weights)
    print(sum(weights))
    for l in range(0, n):
        summ = 0
        for k in range(1, n + 1):
            summ += probability_best_chosen_BR(n, l, k) * weights[k-1]
        l_probs[l] = summ
    return l_probs


""" This code shows what the best response of each candidate is"""
# n = 10
# cols = 2
# l = 9
# best_choices = np.zeros((n, cols))
# for i in range(0, n):
#     k = i + 1
#     if 1 <= k < (n - l + 1):
#         if l < (math.factorial(n - l - 1) * math.factorial(n - k)) / (
#                 math.factorial(n - 2) * math.factorial(n - k - l)):
#             best_choices[i, 0] = 1
#         elif l >= (math.factorial(n - l - 1) * math.factorial(n - k)) / (
#                 math.factorial(n - 2) * math.factorial(n - k - l)):
#             best_choices[i, 0] = (n - l)
#     elif k >= (n - l + 1):
#         best_choices[i, 0] = (n - l)
#     else:
#         print("We got an error")
#     best_choices[i, 1] = probability_best_chosen_BR(n, l, k)


# print(best_choices)


def table_it(table):
    index = []
    for a in range(0, n):
        current_k = str(a + 1)
        index.append("k = " + current_k)

    df = pd.DataFrame(table, columns=['Best alpha', 'Probability picking best'], index=index)
    return df


" This is for printing the best responses and employer utility"

# df = table_it(best_choices)
# print(df)
# print(df.to_latex(index=True))

" Here will be the plot for fixed n and vary l. should be quadratic"
" so we get an array with employer utility and l"

n = 50  # This could change


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


# plotable = varied_l(n)
# print(plotable)
#
# # Create a new figure and an axes
# fig, ax = plt.subplots()
#
# # Scatter plot
# ax.scatter(plotable[:, 0], plotable[:, 1], label='Data points')
#
# # Setting the limit for x and y axis
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
#
# # Label x and y axis
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
#
# # Title for the plot
# ax.set_title('Scatter plot of x and y values')
#
# # Displaying the legend
# ax.legend()
#
# # Display the plot
# plt.show()


# n = 4
# probs = best_l(n)
# print("here is probbs")
# print(probs)
# print(sum(probs))
# print(probs / n)
# print(sum(probs / n))
# # max value of probs
# max_value = np.max(probs)
# print(max_value)
#
# print("here is the agr stuff")
# print(np.argmax(probs))
# print(np.max(probs))
# bestL = np.where(probs == max_value)
#
#
def many_n(n_max):
    output = np.zeros((n_max+1, 3))
    # best_l = np.zeros(n_max)
    for n in range(1, n_max+1):
        l_probs = best_l(n)
        print("This is l_probs")
        print(l_probs)
        # probabilities = l_probs / n
        # print(probabilities)
        # print(sum(probabilities))
        # print("NOW IS ADJUSTED")
        # adjusted_probabilities = uniform_weight(l_probs)
        # adjusted_probabilities = linear_decreasing_weight(l_probs)
        # adjusted_probabilities = exponential_decreasing_weight(l_probs)
        # adjusted_probabilities = geometric_decreasing_weight(l_probs, 0.8)
        # adjusted_probabilities = harmonic_decreasing_weight(l_probs)
        # adjusted_probabilities = decaying_decreasing_weight(l_probs, 0.7)
        # initial_amount = 1.0
        # decay_constant = 0.1
        # time_points = np.linspace(0, 10, 10)
        # adjusted_probabilities = radioactive_decay(l_probs, )

        # print(adjusted_probabilities)
        # print(sum(adjusted_probabilities))
        max_value = np.max(l_probs)
        best_l_2 = np.argmax(l_probs)
        print("This is max value")
        print(max_value)
        print("This is best l_2")
        print(best_l_2)
        output[n, 0] = n
        output[n, 1] = best_l_2 / n
        output[n, 2] = max_value

    return output


test = np.ones(10)
print(test)


def uniform_weight(cumulative_probability):
    return cumulative_probability / len(cumulative_probability)


# print("Uniform")
# print(uniform_weight(test))
# print(sum(uniform_weight(test)))

" Not working "


def linear_decreasing_weight(cumulative_probability):
    adjusted_probability = np.zeros(len(cumulative_probability))
    for i in range(0, len(cumulative_probability)):
        adjusted_probability[i] = 2 * cumulative_probability[i] * ((len(cumulative_probability) - (i)) / (
                len(cumulative_probability) * (len(cumulative_probability) + 1)))
    return adjusted_probability


# print("linear decreasing")
# print(linear_decreasing_weight(test))
# print(sum(linear_decreasing_weight(test)))


def exponential_decreasing_weight(cumulative_probability):
    adjusted_probability = np.zeros(len(cumulative_probability))
    total_weight = 0
    for i in range(0, len(cumulative_probability)):
        adjusted_probability[i] = (cumulative_probability[i] * 2 ** (len(cumulative_probability) - (i))) / (
                (2 ** ((len(cumulative_probability) + 1))) - 1)
        total_weight += adjusted_probability[i]

    adjusted_probability /= total_weight

    return adjusted_probability


# print("exponential decreasing")
# print(exponential_decreasing_weight(test))
# print(sum(exponential_decreasing_weight(test)))


def geometric_decreasing_weight(cumulative_probability, r):
    a = (1 - r) / (1 - r ** (len(cumulative_probability)))
    adjusted_probability = np.zeros(len(cumulative_probability))
    for i in range(0, len(cumulative_probability)):
        # adjusted_probability[i] = (cumulative_probability[i] * a * r ** (len(cumulative_probability) - (i + 1)))
        adjusted_probability[i] = cumulative_probability[i] * a * r ** i
    return adjusted_probability


# print("geometric decreasing")
# print(geometric_decreasing_weight(test, .5))
# print(sum(geometric_decreasing_weight(test, .5)))


def harmonic_decreasing_weight(cumulative_probability):
    n = len(cumulative_probability)
    adjusted_probability = np.zeros(n)
    harmonic_sum = sum(1 / (i + 1) for i in range(n))
    for i in range(n):
        adjusted_probability[i] = cumulative_probability[i] / harmonic_sum / (i + 1)

    return adjusted_probability


# print("harmonic decreasing")
# print(harmonic_decreasing_weight(test))
# print(sum(harmonic_decreasing_weight(test)))


def decaying_decreasing_weight(cumulative_probability, decay_constant):
    n = len(cumulative_probability)
    adjusted_probability = np.zeros(n)

    for i in range(n):
        adjusted_probability[i] = np.exp(-decay_constant * i)

    return adjusted_probability / sum(adjusted_probability)


# print("decaying decreasing")
# print(decaying_decreasing_weight(test, 0.7))
# print(sum(decaying_decreasing_weight(test, 0.7)))


output = many_n(50)
# print(output)
output = output[1:]
print(output)
#
weights = np.ones(50)
# weights = uniform_weight(weights)
# weights = linear_decreasing_weight(weights)
# weights = exponential_decreasing_weight(weights)
# weights = geometric_decreasing_weight(weights, 0.8)
# weights = harmonic_decreasing_weight(weights)
weights = decaying_decreasing_weight(weights, 0.5)

print(weights)
# print(sum(weights))

#
# # n = 10  # Length of the array
# # output = np.zeros((n, 3))  # Your 3-dimensional NumPy array
# #
# # # Generate random data for demonstration purposes
# # output[:, 0] = np.linspace(0, 1, n)  # Assign values to the first column (x-axis)
# # output[:, 1] = np.random.random(n)  # Assign random values to the second column (blue points)
# # output[:, 2] = np.random.random(n)  # Assign random values to the third column (red points)
#
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
#
# # folder_path = "C:\\Users\\Tom McGrath\\Desktop\\TempUni\\Master\\Thesis\\CompetitiveSecretaryProblem\\Plots"
# # filename = "plot1.png"
# # plt.savefig((folder_path + '\\' + filename))
# import math
#
print(1 / math.e)

# import Classic as classic

# output1 = classic.multiple_n(50)
# output1 = output1[1:]
# print(output1)
#
# print("Next is combined")
# combined_output = np.hstack((output1[:, [0, 1, 2]], output[:, 1:]))
# print(combined_output)
#
# # Create the plot
# plt.scatter(combined_output[:, 0], combined_output[:, 1], c='blue', label='Classic Search Fraction')
# plt.scatter(combined_output[:, 0], combined_output[:, 2], c='red', label='Classic Success Rate')
# plt.scatter(combined_output[:, 0], combined_output[:, 3], c='green', label='Competitive Search Fraction')
# plt.scatter(combined_output[:, 0], combined_output[:, 4], c='purple', label='Competitive Success Rate')
#
# plt.axhline(1/np.e, color='gray', linestyle='dotted', alpha=0.3, label='1/e')
# plt.xlabel('Number of Candidates')
# plt.ylabel('Probability/Search Fraction')
# plt.title('Comparison between Secretary Problem')
# plt.legend()
# plt.ylim(-0.01, 1.01)  # Set the y-axis limits to 0 and 1
#
# # Display the plot
# plt.show()
