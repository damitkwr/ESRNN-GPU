# Heavily borrowed from https://github.com/zalandoresearch/pytorch-dilated-rnn

import torch
import torch.nn as nn


class DRNN(nn.Module):
    def __init__(self, inp, hidden, n_layers, out_type, batch_first=False):
        super(DRNN, self).__init__()

        self.dilations = [2 ** i for i in range(n_layers)]
        self.batch_first = batch_first

        self.cells = nn.ModuleList(
            [nn.LSTM(inp, hidden) if i == 0 else nn.LSTM(hidden, hidden) for i in range(n_layers)])

        self.out_type = out_type
        assert self.out_type in ['stack', 'concat']  # Make sure its like stack or concat

    def base_layer(self, inp, lstm_cell, rate):
        steps = len(inp)
        b_size = inp[0].size()[0]
        h_size = lstm_cell.hidden_size

        new_inp, dilate_steps = self.pad_inp(inp, steps, rate)
        if self.out_type == 'stack':
            dilated_inp = torch.stack([new_inp[i::rate, :, :] for i in range(rate)], 2)
        else:
            dilated_inp = torch.cat([new_inp[i::rate, :, :] for i in range(rate)], 1)

    def forward(self, inp):
        inp = inp.transpose(0, 1) if self.batch_first else inp

        out = []

    def pad_inp(self, inp, steps, rate):

        # Checking to see if we need to add to make it even
        if steps % rate != 0:
            dilate_steps = steps // rate + 1
            pad = torch.zeros(dilate_steps * rate - inp.size(0), inp.size(1), inp.size(2)).cuda()
            new_inp = torch.cat((inp, torch.autograd.Variable(
                pad)))  # have to make it Autograd variable otherwise PyTorch might not calculate gradients
        else:
            dilate_steps = steps // rate
            new_inp = inp

        return new_inp, dilate_steps
