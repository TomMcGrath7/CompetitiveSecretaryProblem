from scipy.special import comb, factorial
import matplotlib.pyplot as plt
import numpy as np

def c_star(l, n):
    # calculates the probability of a picking the best for a given l and n in the classic game, c* would be the l
    # that maximises this expression. If l is 0 then there is just a 1/n chance of picking the best.
    # Otherwise we just do the summation from l+1 to n.
    if l == 0:
        return 1 / n
    else:
        return (l / n) * sum(1 / (i - 1) for i in range(l + 1, n + 1))


def c_star_n_minus_1(l, n):
    # calculates the probability of a picking the best for a given l and n in the classic game where we can ignore
    # the last player, c*(n-1) would be the l that maximises this expression
    # If l is 0 then there is just a 1/n chance of picking the best. Otherwise we just do the summation from l+1 to n-1.
    # Since we can basically ignore the last player.
    if n == 1:
        return 1
    elif l == 0:
        return 1 / (n-1)
    else:
        return (l / (n - 1)) * sum(1 / (i - 1) for i in range(l + 1, n))


def l_star(l, n, k):
    # calculates the probability of picking the best player in the competitive game for a given l,n and k
    if k == 1:
        return 1
    elif k != 1 and l == 0:
        return 0
    # threshold = (factorial(n - l - 1) * factorial(n - k)) / (factorial(n - 2) * factorial(n - k - l))
    if k >= (n - l + 1):
        summation = 0
        for i in range(l + 1, n):
            summation += 1 / (i - 1)
        return (l / (n - 1)) * summation
    elif l < (factorial(n - l - 1) * factorial(n - k)) / (factorial(n - 2) * factorial(n - k - l)):
        summation = 0
        for i in range(2, k):
            summation += comb(n - (i + 1), l - 1) / (i - 1)
        return (factorial(l) * factorial(n - l - 1) / factorial(n - 1)) * summation
    else:
        summation = 0
        for i in range(l + 1, n):
            summation += 1 / (i - 1)
        return (l / (n - 1)) * summation


# define empty array for c* values
c_star_values = []
c_star_n_minus_1_values = []
l_star_values = []
e = np.e

# We loop through all values of n from 1 to 130 and calculate the best l for each n
# We append these best values to the array where we store the best l for each optimisation function.
# We set the best value to 0 and the best l to 0 each time we start a new n.
# We then loop through all values of l from 0 to n and calculate the best l for each n.
# For the competive problem we need to also sum over each k from 1 to n.

for n in range(1, 20):
    best_c_star_value = 0
    best_c_star_n_minus_1_value = 0
    best_l_star_value = 0
    current_best_l_c_star = 0
    current_best_l_c_star_n_minus_1 = 0
    current_best_l_l_star = 0
    for l in range(0, n):
        if c_star(l, n) > best_c_star_value:
            best_c_star_value = c_star(l, n)
            current_best_l_c_star = l
        if c_star_n_minus_1(l, n) > best_c_star_n_minus_1_value:
            best_c_star_n_minus_1_value = c_star_n_minus_1(l, n)
            current_best_l_c_star_n_minus_1 = l
        sum_current_l = 0
        for k in range(1, n + 1):
            sum_current_l += l_star(l, n, k)
        if sum_current_l > best_l_star_value:
            best_l_star_value = sum_current_l
            current_best_l_l_star = l
    c_star_values.append(current_best_l_c_star)
    c_star_n_minus_1_values.append(current_best_l_c_star_n_minus_1)
    l_star_values.append(current_best_l_l_star)

print(c_star_values)
print(c_star_n_minus_1_values)
print(l_star_values)

# We divide the values by the index + 1 to get the search fraction for each n.

c_star_plot_values = c_star_values / (np.arange(len(c_star_values)) + 1)
c_star_n_minus_1_plot_values = c_star_n_minus_1_values / (np.arange(len(c_star_n_minus_1_values)) + 1)
l_star_plot_values = l_star_values / (np.arange(len(l_star_values)) + 1)

# Creating a dot plot
plt.figure(figsize=(10, 6))
plt.plot(c_star_plot_values, 'o', label='C*')
plt.plot(c_star_n_minus_1_plot_values, 'o', label='C* n-1')
plt.plot(l_star_plot_values, 'o', label='L*')

plt.axhline(y=1/e, color='r', linestyle='-', label='1/e')

# Adding legend and labels
plt.legend()
plt.xlabel('Index')
plt.ylabel('Value / (Index+1)')
plt.title('Dot Plot Comparison')

# Show plot
# plt.show()


def create_latex_table(c_star, c_star_n_minus_1, l_star):
    header = "\\begin{table}[h!]\n\\centering\n\\begin{tabular}{|c|c|c|}\n\\hline\n"
    column_titles = "C* & C*-1 & L* \\\\ \\hline\n"
    footer = "\\hline\n\\end{tabular}\n\\caption{Your Caption Here}\n\\label{table:your_label}\n\\end{table}"

    body = ""
    for c, cn, l in zip(c_star, c_star_n_minus_1, l_star):
        row = f"{c} & {cn} & {l} \\\\\n"
        body += row

    return header + column_titles + body + footer


latex_table = create_latex_table(c_star_values, c_star_n_minus_1_values, l_star_values)
print(latex_table)