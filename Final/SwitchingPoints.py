import numpy as np
import math
import matplotlib.pyplot as plt

n = 100
switch_x = np.zeros((n+1, 2))
for k in range(1, n+1):
    for x in range(1, n+1):
        if k >= (n - round(n/x)+1):
            switch_x[k, 0] = k
            switch_x[k, 1] = x
        elif round(n/x) <= (math.factorial(n- round(n/x)-1)/math.factorial(n-2))*(math.factorial(n-k)/(math.factorial(n-k-round(n/x)))):
            switch_x[k, 0] = k
            switch_x[k, 1] = x

print(switch_x)

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



