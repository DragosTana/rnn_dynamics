from dataset import FlipFlopData
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


data_generator = FlipFlopData(n_bits=1, n_time=10000, p=0.5, random_seed=np.random.randint(0, 1000))
dataset = []
for i in range(10):
    dataset.append(data_generator.generate_data(n_trials=1))

class MyRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super(MyRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.rnn = nn.RNN(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, input, prev_state=None):
                
        h0 = torch.zeros(1, 1, self.hidden_size)
        output, hn = self.rnn(input, h0)
        output = self.fc(output)
        output = torch.tanh(output)
        
        return output
    
    
    
rnn = MyRNN(1, 6)
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
            
            losses.append(loss.item())
            
test = data_generator.generate_data(n_trials=1)
inputs = test['inputs']
targets = test['targets']
results = []
for i in range(len(inputs[0])):
    input_data = torch.tensor([[inputs[0][i]]])
    output = rnn.forward(input_data, prev_state)
    prev_state = output
    results.append(output.item())
    
results = np.array(results).reshape(-1, 1)
data_generator._plot_single_trial(inputs[0][:64], targets[0][:64], results[:64])
plt.show()

