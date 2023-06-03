import numpy as np
import matplotlib.pyplot as plt


def prob_best(x):
    return -x * np.log(x)


""" Classic Secretary Problem Probabilities """


def classic_probability(n, l):
    if l == 0:
        return 1 / n
    else:
        sum = 0
        for i in range(l + 1, n + 1):
            sum += (1 / (i - 1))
        return (l / n) * sum


def find_best_l(n):
    best_l = 0
    best_probability = classic_probability(n, best_l)

    for l in range(1, n):
        probability = classic_probability(n, l)
        if probability > best_probability:
            best_l = l
            best_probability = probability

    return best_l


n = 10
best_l = find_best_l(n)
print("Best l for n =", n, "is", best_l)


def multiple_n(n_max):
    output = np.zeros((n_max, 3))
    for n in range(1, n_max):
        best_l = find_best_l(n)
        l_ratio = best_l/n
        prob_of_best = classic_probability(n, best_l)
        output[n, 0] = n
        output[n, 1] = best_l
        output[n, 2] = prob_of_best

    return output


output = multiple_n(13)
# print(output)
output = output[1:]
print(output)



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
plt.show()