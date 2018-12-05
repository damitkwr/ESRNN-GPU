import torch
import torch.nn as nn
from es_rnn.DRNN import DRNN


class ESRNN(nn.Module):
    def __init__(self, num_series, config):
        super(ESRNN, self).__init__()
        self.config = config
        self.num_series = num_series
        self.add_nl_layer = self.config['add_nl_layer']

        init_lev_sms = []
        init_seas_sms = []
        init_seasonalities = []

        for i in range(num_series):
            init_lev_sms.append(nn.Parameter(torch.Tensor([0.5]), requires_grad=True))
            init_seas_sms.append(nn.Parameter(torch.Tensor([0.5]), requires_grad=True))
            init_seasonalities.append(nn.Parameter((torch.ones(config['seasonality']) * 0.5), requires_grad=True))

        self.init_lev_sms = nn.ParameterList(init_lev_sms)
        self.init_seas_sms = nn.ParameterList(init_seas_sms)
        self.init_seasonalities = nn.ParameterList(init_seasonalities)

        self.nl_layer = nn.Linear(config['state_hsize'],
                                  config['state_hsize'])
        self.act = nn.Tanh()
        self.scoring = nn.Linear(config['state_hsize'], config['output_size'])

        self.logistic = nn.Sigmoid()

        self.resid_drnn = ResidualDRNN(self.config)

    def forward(self, train, val, test, info_cat, idxs, testing=False):
        # GET THE PER SERIES PARAMETERS
        lev_sms = self.logistic(torch.stack([self.init_lev_sms[idx] for idx in idxs]).squeeze(1))
        seas_sms = self.logistic(torch.stack([self.init_seas_sms[idx] for idx in idxs]).squeeze(1))
        init_seasonalities = torch.stack([self.init_seasonalities[idx] for idx in idxs])

        seasonalities = []
        # PRIME SEASONALITY
        for i in range(self.config['seasonality']):
            seasonalities.append(torch.exp(init_seasonalities[:, i]))
        seasonalities.append(torch.exp(init_seasonalities[:, 0]))

        if testing:
            train = torch.cat((train, val), dim=1)

        train = train.float()

        levs = []
        log_diff_of_levels = []

        levs.append(train[:, 0] / seasonalities[0])
        for i in range(1, train.shape[1]):
            # CALCULATE LEVEL FOR CURRENT TIMESTEP TO NORMALIZE RNN
            new_lev = lev_sms * (train[:, i] / seasonalities[i]) + (1 - lev_sms) * levs[i - 1]
            levs.append(new_lev)

            # STORE DIFFERENCE TO PENALIZE LATER
            log_diff_of_levels.append(torch.log(new_lev / levs[i - 1]))

            # CALCULATE SEASONALITY TO DESEASONALIZE THE DATA FOR RNN
            seasonalities.append(seas_sms * (train[:, i] / new_lev) + (1 - seas_sms) * seasonalities[i])

        seasonalities_stacked = torch.stack(seasonalities).transpose(1, 0)
        levs_stacked = torch.stack(levs).transpose(1, 0)

        loss_mean_sq_log_diff_level = 0
        if self.config['level_variability_penalty'] > 0:
            sq_log_diff = torch.stack(
                [(log_diff_of_levels[i] - log_diff_of_levels[i - 1]) ** 2 for i in range(1, len(log_diff_of_levels))])
            loss_mean_sq_log_diff_level = torch.mean(sq_log_diff)

        if self.config['output_size'] > self.config['seasonality']:
            start_seasonality_ext = seasonalities_stacked.shape[1] - self.config['seasonality']
            seasonalities_stacked = torch.cat((seasonalities_stacked, seasonalities_stacked[:, start_seasonality_ext:]),
                                              dim=1)

        window_input_list = []
        window_output_list = []
        for i in range(self.config['input_size'] - 1, train.shape[1]):
            input_window_start = i + 1 - self.config['input_size']
            input_window_end = i + 1

            train_deseas_window_input = train[:, input_window_start:input_window_end] / seasonalities_stacked[:,
                                                                                        input_window_start:input_window_end]
            train_deseas_norm_window_input = (train_deseas_window_input / levs_stacked[:, i].unsqueeze(1))
            train_deseas_norm_cat_window_input = torch.cat((train_deseas_norm_window_input, info_cat), dim=1)
            window_input_list.append(train_deseas_norm_cat_window_input)

            output_window_start = i + 1
            output_window_end = i + 1 + self.config['output_size']

            if i < train.shape[1] - self.config['output_size']:
                train_deseas_window_output = train[:, output_window_start:output_window_end] / \
                                             seasonalities_stacked[:, output_window_start:output_window_end]
                train_deseas_norm_window_output = (train_deseas_window_output / levs_stacked[:, i].unsqueeze(1))
                window_output_list.append(train_deseas_norm_window_output)

        window_input = torch.cat([i.unsqueeze(0) for i in window_input_list], dim=0)
        window_output = torch.cat([i.unsqueeze(0) for i in window_output_list], dim=0)

        self.train()
        network_pred = self.series_forward(window_input[:-self.config['output_size']])
        network_act = window_output

        self.eval()
        network_output_non_train = self.series_forward(window_input)

        # USE THE LAST VALUE OF THE NETWORK OUTPUT TO COMPUTE THE HOLDOUT PREDICTIONS
        hold_out_output_reseas = network_output_non_train[-1] * seasonalities_stacked[:, -self.config['output_size']:]
        hold_out_output_renorm = hold_out_output_reseas * levs_stacked[:, -1].unsqueeze(1)

        # WE KNOW THE DATA IS STRICTLY POSITIVE
        hold_out_pred = hold_out_output_renorm * torch.gt(hold_out_output_renorm, 0).float()
        hold_out_act = test if testing else val

        self.train()
        # RETURN JUST THE TRAINING INPUT RATHER THAN THE ENTIRE SET BECAUSE THE HOLDOUT IS BEING GENERATED WITH THE REST
        return network_pred, network_act, hold_out_pred, hold_out_act, loss_mean_sq_log_diff_level

    def series_forward(self, data):
        data = self.resid_drnn(data)
        if self.add_nl_layer:
            data = self.nl_layer(data)
            data = self.act(data)
        data = self.scoring(data)
        return data


class ResidualDRNN(nn.Module):
    def __init__(self, config):
        super(ResidualDRNN, self).__init__()
        self.config = config

        layers = []
        for grp_num in range(len(self.config['dilations'])):

            if grp_num == 0:
                input_size = self.config['input_size'] + self.config['num_of_categories']
            else:
                input_size = self.config['state_hsize']

            l = DRNN(input_size,
                     self.config['state_hsize'],
                     n_layers=len(self.config['dilations'][grp_num]),
                     dilations=self.config['dilations'][grp_num],
                     cell_type=self.config['rnn_cell_type'])

            layers.append(l)

        self.rnn_stack = nn.Sequential(*layers)

    def forward(self, input_data):
        for layer_num in range(len(self.rnn_stack)):
            residual = input_data
            out, _ = self.rnn_stack[layer_num](input_data)
            if layer_num > 0:
                out += residual
            input_data = out
        return out
