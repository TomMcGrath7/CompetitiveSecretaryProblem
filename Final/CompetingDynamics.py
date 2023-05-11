import numpy as np
import math
import itertools

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
        if l <= (math.factorial(n - l - 1) * math.factorial(n - k)) / (math.factorial(n - 2) * math.factorial(n - k - l)):  # confirm the right bound
            numerator = 0
            for i in range(2, k):
                numerator += (1 / (i - 1)) * ((math.factorial(l) * math.factorial(n - l - 1) * math.factorial(n - i - 1)) /
                                               (math.factorial(l - 1) * math.factorial(n - l - i)))
            denominator = math.factorial(n - 1)
            return numerator / denominator

        elif l > (math.factorial(n - l - 1) * math.factorial(n - k)) / (math.factorial(n - 2) * math.factorial(n - k - l)):
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


def best_l(n):
    l_probs = np.zeros(n)
    for l in range(0,n):
        sum = 0
        for k in range(1,n+1):
            sum += probability_best_chosen_BR(n, l, k)
        l_probs[l] = sum
    return l_probs


n = 4
probs = best_l(n)
print("here is probbs")
print(probs)
print(sum(probs))
print(probs/n)
print(sum(probs/n))
# max value of probs
max_value = np.max(probs)
print(max_value)

print("here is the agr stuff")
print(np.argmax(probs))
print(np.max(probs))
bestL = np.where(probs == max_value)


def many_n(n_max):
    output = np.zeros((n_max,3))
    # best_l = np.zeros(n_max)
    for n in range(1,n_max):
        l_probs = best_l(n)
        probabilities = l_probs/n
        max_value = np.max(probabilities)
        best_l_2 = np.argmax(probabilities)
        output[n,0] = n
        output[n,1] = best_l_2/n
        output[n,2] = max_value

    return output


output = many_n(50)
print(output)
output = output[1:]
print(output)

import matplotlib.pyplot as plt

# n = 10  # Length of the array
# output = np.zeros((n, 3))  # Your 3-dimensional NumPy array
#
# # Generate random data for demonstration purposes
# output[:, 0] = np.linspace(0, 1, n)  # Assign values to the first column (x-axis)
# output[:, 1] = np.random.random(n)  # Assign random values to the second column (blue points)
# output[:, 2] = np.random.random(n)  # Assign random values to the third column (red points)

# Create the plot
plt.scatter(output[:, 0], output[:, 1], c='blue', label='Search Fraction')
plt.scatter(output[:, 0], output[:, 2], c='red', label='Probability Picking best')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Plot of Points')
plt.legend()
plt.ylim(0, 1)  # Set the y-axis limits to 0 and 1

# Display the plot
# plt.show()

folder_path = "C:\\Users\\Tom McGrath\\Desktop\\TempUni\\Master\\Thesis\\CompetitiveSecretaryProblem\\Plots"
filename = "plot1.png"
plt.savefig((folder_path + '\\' + filename))


