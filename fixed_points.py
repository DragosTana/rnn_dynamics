import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root, brenth, root_scalar
from scipy.optimize import newton
import time
import math
import copy
def function(x, b):
    return 5*np.tanh(x) - x + b

def function2(x, w):
    return w*np.tanh(x) - x

def derivative(x, b):
    return -1 + 5 * (1 - np.tanh(x)**2)

def derivative2(x, w):
    return -1 + w * (1 - np.tanh(x)**2)
    
fig, ax = plt.subplots()
b = np.linspace(-10, 10, 1000)
for i in range(len(b)):
    initializations = np.linspace(-10, 10, 3)
    for j in range(len(initializations)):
        #x = root_scalar(function, x0=initializations[j],fprime=derivative, bracket=[-20,20], args=(b[i]), method='bisect').root
        x = fsolve(function, initializations[j], fprime=derivative, args=(b[i]))
        if not math.isclose(abs(x), 1.444, abs_tol=0.01):
            if derivative(x, b[i]) < 0:
                stable_points = ax.scatter(b[i], x, c='g')
            else:
                unstable_points = ax.scatter(b[i], x, c='r')

plt.legend((stable_points, unstable_points), ('stable', 'unstable'))
plt.ylabel('Equilibrium points')
plt.xlabel('W_rec')
plt.show()