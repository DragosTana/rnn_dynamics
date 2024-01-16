from dataset import FlipFlopData
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


data_generator = FlipFlopData(n_bits=1, n_time=10, p=0.5, random_seed=2)
train_data = data_generator.generate_data(n_trials=10)
test_data = data_generator.generate_data(n_trials=1)

class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyRNN, self).__init__()

        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])

        return out
    
rnn = MyRNN(1, 1, 1)
optimizer = optim.SGD(rnn.parameters(), lr=0.01)
critirion = torch.nn.MSELoss()
epochs = 100



