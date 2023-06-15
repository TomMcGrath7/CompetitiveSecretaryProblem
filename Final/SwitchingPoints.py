import numpy as np
import math
import matplotlib.pyplot as plt
#
# n = 10
# switch_x = np.zeros((n+1, 2))
# for k in range(1, n+1):
#     for x in range(10, (n*10)+1):
#         value = x/10
#         print(round(n/value))
#         if k >= (n - round(n/value)+1):
#             if switch_x[k, 0] == 0:
#                 switch_x[k, 0] = k
#                 switch_x[k, 1] = value
#         elif round(n/value) <= (math.factorial(n- round(n/value)-1)/math.factorial(n-2))*(math.factorial(n-k)/(math.factorial(n-k-round(n/value)))):
#             if switch_x[k, 0] == 0:
#                 switch_x[k, 0] = k
#                 switch_x[k, 1] = value
#
# print(switch_x)



n = 100
switch_x = np.zeros((n+1, 2))
for k in range(1, n+1):
    for l in range(0, n):
        if k >= (n - l + 1):
            if switch_x[k, 0] == 0:
                switch_x[k, 0] = k
                switch_x[k, 1] = l/n
        elif l >= (math.factorial(n- l-1)/math.factorial(n-2))*(math.factorial(n-k)/(math.factorial(n-k-l))):
            if switch_x[k, 0] == 0:
                switch_x[k, 0] = k
                switch_x[k, 1] = l/n



switch_x = switch_x[1:]
print(switch_x)

# Create the plot
plt.scatter(switch_x[:, 0], switch_x[:, 1], c='blue', label='Switching Point')
# plt.scatter(switch_x[:, 0], switch_x[:, 2], c='red', label='Probability Picking best')
plt.axhline(1 / np.e, color='gray', linestyle='dotted', alpha=0.3, label='1/e')
plt.xlabel('Candidate k')
plt.ylabel('Switching Fraction')
plt.title('Switching Points for Candidates k')
plt.legend()
plt.ylim(-0.03, 1.03)

# Add the faint line for 'weights' across the x-axis
# plt.plot(output[0:, 0], weights, color='gray', linewidth=0.5, alpha=0.5)
# Display the plot
plt.show()



