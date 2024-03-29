from dataset import FlipFlopData
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from torch.autograd import Variable

class RNNCell(nn.Module):
    
    def __init__(self, input_size, hidden_size, nonlinearity='tanh'):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        if self.nonlinearity not in ["tanh", "sigmoid"]:
            raise ValueError("Invalid nonlinearity selected for RNN.")
        
        #self.W_rec = torch.nn.Parameter(torch.tensor([5.0]), requires_grad=True)
        #self.W_in = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=False)
        #self.bias = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        
        self.W_rec = torch.nn.Parameter(torch.randn((hidden_size, hidden_size)), requires_grad=True)
        self.W_in = torch.nn.Parameter(torch.randn((hidden_size, input_size)), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.randn((hidden_size)).T, requires_grad=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, input, prev_state=None):
        
        if prev_state is None:
            prev_state = torch.zeros(self.hidden_size, 1)
        else:
            prev_state = Variable(prev_state)
            
        input = Variable(input).view(1)
        
        if self.nonlinearity == "tanh":
            next_state = torch.add(torch.add(torch.matmul(self.W_rec, torch.tanh(prev_state)).view(self.hidden_size), torch.matmul(self.W_in, input)), self.bias)
        elif self.nonlinearity == "sigmoid":
            next_state = torch.add(torch.add(torch.matmul(self.W_rec, torch.sigmoid(prev_state)).view(self.hidden_size), torch.matmul(self.W_in, input)), self.bias)
        
        output = self.fc(next_state)
        return (output, next_state)

data_generator = FlipFlopData(n_bits=1, n_time=1000, p=0.5, random_seed=10)
dataset = []
for i in range(20):
    dataset.append(data_generator.generate_data(n_trials=1))

def function(x, w, b):
    return w*np.tanh(x) - x + b

def derivative(x, w, b):
    return -1 + 5 * (1 - np.tanh(x)**2)

rnn = RNNCell(input_size=1, hidden_size=10, nonlinearity='tanh')

optimizer = optim.Adam(rnn.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
critirion = torch.nn.MSELoss()
epochs = 1
losses = []
prev_state = torch.zeros((rnn.hidden_size, 1))

fig, ax = plt.subplots()

for i in range(epochs):
    print("Epoch: ", i)
    for j in range(len(dataset)):
        
        inputs = dataset[j]['inputs']
        targets = dataset[j]['targets']
        for k in range(len(inputs[0])):
            input_data = torch.tensor([[inputs[0][k]]])
            target_data = torch.tensor([[targets[0][k]]])
            output, next_state = rnn.forward(input_data, prev_state)
            prev_state = next_state
            loss = critirion(output, target_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            ########## PLOTTING ############
            #b = np.linspace(-10, 10, 100)
            #points = []
            #for i in range(len(b)):
            #    initializations = np.linspace(-10, 10, 3)
            #    for j in range(len(initializations)):
            #        root = fsolve(function, initializations[j], fprime=derivative, args=(float(rnn.W_rec.data), b[i]))
            #        points.append((b[i], root))
            #
            #stable_points = [p for p in points if derivative(p[1], rnn.W_rec.data, p[0]) > 1]
            #unstable_points = [p for p in points if derivative(p[1], rnn.W_rec.data, p[0]) < 1]
            #graf_stable = ax.scatter(*zip(*stable_points), c = 'g', s=10)
            #graf_unstable = ax.scatter(*zip(*unstable_points), c = 'r', s=10)
            #status = ax.scatter(rnn.bias.data, output.data, c = 'b', s=40)
            #
            #plt.pause(0.01)
            #graf_stable.remove()
            #graf_unstable.remove()
            #status.remove()
            ################################
            
            losses.append(loss.data)
            #scheduler.step(loss)
            
test_data = data_generator.generate_data(n_trials=1)
inputs = test_data['inputs']
targets = test_data['targets']
prev_state = torch.zeros((rnn.hidden_size, 1))

results = []
for i in range(len(inputs[0])):
    input_data = torch.tensor([[inputs[0][i]]])
    output, next_state = rnn.forward(input_data, prev_state)
    prev_state = next_state
    results.append(output.item())
    
results = np.array(results).reshape(-1, 1)
print(inputs[0].shape, targets[0].shape, results.shape)

data_generator._plot_single_trial(inputs[0][:100], targets[0][:100], results[:100])
plt.legend()
plt.show()

