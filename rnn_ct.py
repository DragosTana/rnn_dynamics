import numpy as np
import matplotlib.pyplot as plt

# Define the RNN dynamics (you can customize this based on your specific RNN cell)
def rnn_dynamics(t, y):
    # y[0] represents the state of the RNN cell
    # Modify the following line based on your RNN dynamics
    dydt = -y[0] + 5 * np.tanh(y[0]) + 3.1706
    return dydt

# Runge-Kutta integration method
def runge_kutta(f, y0, t_span, h):
    t_values = np.arange(t_span[0], t_span[1] + h, h)
    num_steps = len(t_values)
    y_values = np.zeros((num_steps, len(y0)))
    y_values[0] = y0

    for i in range(1, num_steps):
        k1 = h * f(t_values[i - 1], y_values[i - 1])
        k2 = h * f(t_values[i - 1] + 0.5 * h, y_values[i - 1] + 0.5 * k1)
        k3 = h * f(t_values[i - 1] + 0.5 * h, y_values[i - 1] + 0.5 * k2)
        k4 = h * f(t_values[i - 1] + h, y_values[i - 1] + k3)

        y_values[i] = y_values[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return t_values, y_values

# Initial conditions
initial_state = np.array([0.0])

# Time span and step size
time_span = (0, 10)
step_size = 0.1

# Run the simulation
t_values, y_values = runge_kutta(rnn_dynamics, initial_state, time_span, step_size)

# Plot the results
plt.plot(t_values, y_values[:, 0], label='RNN Cell State')
plt.xlabel('Time')
plt.ylabel('RNN Cell State')
plt.title('Continuous-Time Evolution of RNN Cell')
plt.legend()
plt.show()