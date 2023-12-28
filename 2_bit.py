from model import RNNCell
from dataset import FlipFlopData
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

data_generator = FlipFlopData(n_bits=1, n_time=2000, p=0.5, random_seed=0)
data = data_generator.generate_data(n_trials=1)
inputs = data['inputs']
targets = data['targets']

rnn = RNNCell(1, 1)

optimizer = optim.SGD(rnn.parameters(), lr=0.01)
critirion = torch.nn.MSELoss()

losses = []
prev_state = torch.tensor([[0.0]])

for i in range(len(inputs[0])):
    optimizer.zero_grad()
    input_data = torch.tensor([[inputs[0][i]]])
    target_data = torch.tensor([[targets[0][i]]])
    output = rnn.forward(input_data, prev_state)
    prev_state = output
    loss = critirion(output, target_data)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
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