import numpy as np
import matplotlib.pyplot as plt

output = np.load('output.npy')
print(output)

plt.scatter(output[:, 0], output[:, 1], c='blue', s=0.5, label='Search Fraction')
plt.scatter(output[:, 0], output[:, 2], c='red', s=0.5, label='Probability Picking best')
plt.axhline(1 / np.e, color='gray', linestyle='dotted', alpha=0.3, label='1/e')
plt.xlabel('Number of Candidates')
plt.ylabel('Probability/Search Fraction')
plt.title('Decaying Decreasing Secretary Problem')
plt.legend()
plt.ylim(-0.03, 1.03)  # Set the y-axis limits to 0 and 1

# Add the faint line for 'weights' across the x-axis
# plt.plot(output[0:, 0], weights, color='gray', linewidth=0.5, alpha=0.5)
# Display the plot
plt.show()