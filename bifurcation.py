import numpy as np
import matplotlib.pyplot as plt

# Define the logistic map function
def logistic_map(x, r):
    return r * x * (1 - x)

# Bifurcation diagram function
def bifurcation_diagram(r_values, x0, num_iterations):
    result = []
    for r in r_values:
        x = x0
        for _ in range(num_iterations):
            x = logistic_map(x, r)
            result.append((r, x))
    return np.array(result)

# Set parameters
r_values = np.linspace(2.5, 4.0, 1000)
x0 = 0.5
num_iterations = 1000

# Generate bifurcation diagram data
data = bifurcation_diagram(r_values, x0, num_iterations)

# Plot the bifurcation diagram
plt.scatter(data[:, 0], data[:, 1], s=0.1, edgecolors='none')
plt.xlabel('r')
plt.ylabel('x')
plt.title('Bifurcation Diagram')
plt.show()
