from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.linear_x = nn.Linear(input_dim, hidden_dim)
        # print("linear_x: ", input_dim, hidden_dim)
        self.linear_h = nn.Linear(hidden_dim, input_dim)
        self.tanh = nn.Tanh()
        self.linear_y = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax()


    def forward(self, x):
        # Implementation here ...
        h = torch.zeros((self.batch_size, self.hidden_dim))
        y = None
        for i in range(self.seq_length):
            # print("x: ", x[i].size())
            sample = torch.reshape(x[i], (self.batch_size, 1))
            x_output = self.linear_x(sample)
            # print("x_output: ", (x_output))
            h_output = self.linear_h(h)
            # print("h_output: ", h_output)
            tanh_input = x_output + h_output
            # print("tanh_input: ", tanh_input)
            h = self.tanh(tanh_input)
            # print("h: ", h)
            if i == self.seq_length - 1:
                y_input = self.linear_y(h)
                y = self.softmax(y_input)
        return y


        
    # add more methods here if needed
