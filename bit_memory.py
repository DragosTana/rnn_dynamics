from model import RNNCell
from dataset import FlipFlopData
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

data_generator = FlipFlopData(n_bits=1, n_time=1000, p=0.5, random_seed=np.random.randint(0, 1000))
dataset = []
for i in range(10):
    dataset.append(data_generator.generate_data(n_trials=1))

def function(x, w, b):
    return w*np.tanh(x) - x + b

def derivative(x, w, b):
    return -1 + 5 * (1 - np.tanh(x)**2)

rnn = RNNCell(1, 1)
rnn.W_rec.data = torch.tensor([[5.0]])  

optimizer = optim.SGD(rnn.parameters(), lr=0.01)
critirion = torch.nn.MSELoss()
epochs = 1
losses = []
prev_state = torch.tensor([[0.0]])

fig, ax = plt.subplots()

for i in range(epochs):
    print("Epoch: ", i)
    for j in range(len(dataset)):
        
        inputs = dataset[j]['inputs']
        targets = dataset[j]['targets']
        for k in range(len(inputs[0])):
            input_data = torch.tensor([[inputs[0][k]]])
            target_data = torch.tensor([[targets[0][k]]])
            output = rnn.forward(input_data, prev_state)
            prev_state = output
            loss = critirion(output, target_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            ########## PLOTTING ############
            b = np.linspace(-10, 10, 100)
            points = []
            for i in range(len(b)):
                initializations = np.linspace(-10, 10, 3)
                for j in range(len(initializations)):
                    root = fsolve(function, initializations[j], fprime=derivative, args=(float(rnn.W_rec.data), b[i]))
                    points.append((b[i], root))

            stable_points = [p for p in points if derivative(p[1], rnn.W_rec.data, p[0]) > 1]
            unstable_points = [p for p in points if derivative(p[1], rnn.W_rec.data, p[0]) < 1]
            graf_stable = ax.scatter(*zip(*stable_points), c = 'g', s=10)
            graf_unstable = ax.scatter(*zip(*unstable_points), c = 'r', s=10)
            status = ax.scatter(rnn.bias.data, output.data, c = 'b', s=40)

            plt.pause(0.01)
            graf_stable.remove()
            graf_unstable.remove()
            status.remove()
            ################################
            
            losses.append(loss.data)
            
plt.show()
test_data = data_generator.generate_data(n_trials=1)
inputs = test_data['inputs']
targets = test_data['targets']

results = []
for i in range(len(inputs[0])):
    input_data = torch.tensor([[inputs[0][i]]])
    output = rnn.forward(input_data, prev_state)
    prev_state = output
    results.append(output.item())
    
results = np.array(results).reshape(-1, 1)
print(inputs[0].shape, targets[0].shape, results.shape)

data_generator._plot_single_trial(inputs[0][:64], targets[0][:64], results[:64])
plt.show()

