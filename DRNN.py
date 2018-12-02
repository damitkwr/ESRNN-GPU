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
        assert self.out_type in ['stack', 'concat']  # Make sure its stack or concat

    def base_layer(self, inp, lstm, rate, hidden):
        steps = len(inp)
        b_size = inp[0].size()[0]
        h_size = lstm.hidden_size

        new_inp, dilate_steps = self.pad_inp(inp, steps, rate)
        if self.out_type == 'stack':
            dilated_inp = torch.stack([new_inp[i::rate, :, :] for i in range(rate)], 2)
        else:
            dilated_inp = torch.cat([new_inp[i::rate, :, :] for i in range(rate)], 1)

        if hidden is None:
            # h = torch.autograd.Variable(torch.zeros(b_size, h_size)).cuda()
            # mem = torch.autograd.Variable(torch.zeros(b_size, h_size)).cuda()
            h = torch.autograd.Variable(torch.zeros(dilated_inp.size()[1], h_size))
            mem = torch.autograd.Variable(torch.zeros(dilated_inp.size()[1], h_size))
            hidden = (h.unsqueeze(0), mem.unsqueeze(0))
            dilated_out, hidden = lstm(dilated_inp, hidden)
        else:
            hidden = torch.cat([hidden[i::rate, :, :] for i in range(rate)], 1)
            dilated_out, hidden = lstm(dilated_inp, (hidden, hidden))

        split_outputs = self._split_outputs(dilated_out, rate)

        return split_outputs[:steps], hidden

    def forward(self, inp, hidden=None):
        inp = inp.transpose(0, 1) if self.batch_first else inp

        out = []

        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            if hidden is None:
                inp, _ = self.base_layer(inp, cell, dilation, None)
            else:
                inp, hidden[i] = self.base_layer(inp, cell, dilation, hidden[i])

            out.append(inp[-dilation:])

        if self.batch_first:
            return inp.transpose(0, 1), out
        else:
            return inp, out

    def pad_inp(self, inp, steps, rate):

        # Checking to see if we need to add to make it even
        if steps % rate != 0:
            dilate_steps = steps // rate + 1
            # pad = torch.zeros(dilate_steps * rate - inp.size(0), inp.size(1), inp.size(2)).cuda()
            pad = torch.zeros(dilate_steps * rate - inp.size(0), inp.size(1), inp.size(2))
            new_inp = torch.cat((inp, torch.autograd.Variable(
                pad)))  # have to make it Autograd variable otherwise PyTorch might not calculate gradients
        else:
            dilate_steps = steps // rate
            new_inp = inp

        return new_inp, dilate_steps

    # Directly taken from https://github.com/zalandoresearch/pytorch-dilated-rnn
    def _split_outputs(self, dilated_outputs, rate):
        batchsize = dilated_outputs.size(1) // rate

        blocks = [dilated_outputs[:, i * batchsize: (i + 1) * batchsize, :] for i in range(rate)]

        interleaved = torch.stack((blocks)).transpose(1, 0).contiguous()
        interleaved = interleaved.view(dilated_outputs.size(0) * rate,
                                       batchsize,
                                       dilated_outputs.size(2))
        return interleaved


if __name__ == '__main__':
    n_inp = 10
    n_hidden = 16
    n_layers = 3

    model = DRNN(n_inp, n_hidden, n_layers, 'concat')

    test_x1 = torch.autograd.Variable(torch.randn(26, 2, n_inp))
    test_x2 = torch.autograd.Variable(torch.randn(26, 2, n_inp))

    out, hidden = model(test_x1)
    out, hidden = model(test_x2, hidden)
