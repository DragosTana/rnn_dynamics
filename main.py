import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import copy

from scipy.optimize import fsolve
import time
import math

class RNNCell(nn.Module):
    
    def __init__(self, input_size, hidden_size, nonlinearity='tanh'):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        if self.nonlinearity not in ["tanh", "sigmoid"]:
            raise ValueError("Invalid nonlinearity selected for RNN.")
        
        self.W_rec = torch.nn.Parameter(torch.tensor([5.0]), requires_grad=True)
        self.W_in = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True)


    def forward(self, input, prev_state=None):
        
        if prev_state is None:
            prev_state = Variable(input.new_zeros(input.size(0), self.hidden_size))
        else:
            prev_state = Variable(torch.tensor([prev_state]))
            
        input = Variable(torch.tensor([input]))

        if self.nonlinearity == "tanh":
            next_state = torch.matmul(self.W_rec, torch.tanh(prev_state)) + torch.matmul(self.W_in, input) + self.bias
        elif self.nonlinearity == "sigmoid":
            next_state = torch.matmul(self.W_rec, torch.sigmoid(prev_state)) + torch.matmul(self.W_in, input) + self.bias
        
        return next_state 
    
def function(x, w, b):
    return w*np.tanh(x) - x + b

def derivative(x, w, b):
    return -1 + 5 * (1 - np.tanh(x)**2)

if __name__ == "__main__":
        
        rnn = RNNCell(1, 1)
        for name, param in rnn.named_parameters():
            print(name, param.shape, param.data)
        
        optimizer = optim.SGD(rnn.parameters(), lr=0.01)
        critirion = torch.nn.MSELoss()
        
        input_data = torch.tensor([[0.0]])
        target_data = torch.tensor([[0.8]])
        
        losses = []
        biases = []
        prev_state = torch.tensor([[0.2]])
        
        fig, ax = plt.subplots()
        
        for step in range(1000):
            pred = rnn.forward(input_data, prev_state)
            prev_state = pred
            loss = critirion(pred, target_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            biases.append(copy.deepcopy(rnn.bias.data))
            losses.append(loss.data)
            
            b = np.linspace(-10, 10, 100)
            points = []
            for i in range(len(b)):
                initializations = np.linspace(-10, 10, 3)
                for j in range(len(initializations)):
                    root = fsolve(function, initializations[j], fprime=derivative, args=(rnn.W_rec.data, b[i]))
                    points.append((b[i], root))
                    
            stable_points = [p for p in points if derivative(p[1], rnn.W_rec.data, p[0]) > 1]
            unstable_points = [p for p in points if derivative(p[1], rnn.W_rec.data, p[0]) < 1]
            status = ax.scatter(rnn.bias.data, pred.data, c = 'g', s=40)
            graf_stable = ax.scatter(*zip(*stable_points), c = 'b', s=10)
            graf_unstable = ax.scatter(*zip(*unstable_points), c = 'r', s=10)
            
            plt.legend((status, graf_stable, graf_unstable),
                        ('Current state', 'Stable points', 'Unstable points'),
                        scatterpoints=1,
                        loc='lower left',
                        ncol=3,
                        fontsize=10)
            
            plt.pause(0.001)
            status.remove()
            graf_stable.remove()
            graf_unstable.remove()

        plt.show()
