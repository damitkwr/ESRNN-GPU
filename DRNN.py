# Heavily borrowed from https://github.com/zalandoresearch/pytorch-dilated-rnn

import torch
import torch.nn as nn
import torch.autograd as autograd


class DRNN(nn.Module):
    def __init__(self, inp, hidden, n_layers, batch_first=False):
        super(DRNN, self).__init__()

        self.dilations = [2 ** i for i in range(n_layers)]
        self.batch_first = batch_first

        self.cells = nn.ModuleList(
            [nn.LSTM(inp, hidden) if i == 0 else nn.LSTM(hidden, hidden) for i in range(n_layers)])

    def base_layer(self, inp, lstm_cell, rate):
        steps = len(inp)
        b_size = inp[0].size()[0]
        h_size = lstm_cell.hidden_size

    def forward(self, inp):
        inp = inp.transpose(0, 1) if self.batch_first else inp

        out = []
