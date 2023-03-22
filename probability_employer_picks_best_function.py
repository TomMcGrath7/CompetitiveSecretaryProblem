import numpy as np
import math


def proability_picking_best(n, l, k):
    sum = 0
    for i in range(2, k):
        sum += (math.factorial(l)*math.factorial(n-l-1)*(math.factorial(n-(i+1))/(math.factorial(l-1)*math.factorial(n-i-l-2))))*(1/(i-1))
    denominator = math.factorial(n-1)
    return sum/denominator


