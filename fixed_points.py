import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root, brenth, root_scalar
from scipy.optimize import newton
import time
import math
import copy
def function(x, b):
    return 5*np.tanh(x) - x + b

def derivative(x, b):
    return -1 + 5 * (1 - np.tanh(x)**2)

fig, ax = plt.subplots()
b = np.linspace(-10, 10, 100)
for i in range(len(b)):
    initializations = np.linspace(-10, 10, 3)
    for j in range(len(initializations)):
        #x = root_scalar(function, x0=initializations[j],fprime=derivative, bracket=[-20,20], args=(b[i]), method='bisect').root
        x = fsolve(function, initializations[j], fprime=derivative, args=(b[i]))
        if derivative(x, b[i]) < 0:
            ax.scatter(b[i], x, c='r')
        else:
            ax.scatter(b[i], x, c='b')
    
states = np.random.uniform(-10, 10, 100)
points = [(b[i], states[i]) for i in range(len(states))]
iterations = 1000
for i in range(iterations):
    print("Iteration: ", i) 
    graf_states = ax.scatter(*zip(*points), c = 'g', s=10)
    new_points = [(b[i], function(points[i][1], b[i])) for i in range(len(points))]
    points = new_points
    plt.pause(0.01)
    graf_states.remove()
    
plt.show()