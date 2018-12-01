import torch
import torch.nn as nn
import copy
from utils.helper_funcs import unpad_sequence

class ESRNN(nn.Module):
    def __init__(self, num_series, config):
        super(ESRNN, self).__init__()
        self.config = config
        self.num_series = num_series

        init_lev_sms = []
        init_seas_sms = []
        init_seasonalities = []

        # NEED TO ENSURE THAT THE GRADIENTS ARE ACCRUEING, IF NOT NEED TO TRY CREATING THESE PARAMETERS AS VARIABLES
        # ANOTHER THING TO LOOK AT IS RATHER THAN INDEXING NORMALLY TO USE INDEX_SELECT METHOD ON THE TENSOR
        #             UPDATE 2018-11-30: PARAMETERS SHOWING IN MODEL PRINT (AREDD)
        for i in range(num_series):
            init_lev_sms.append(nn.Parameter(torch.Tensor([0.5])))
            init_seas_sms.append(nn.Parameter(torch.Tensor([0.5])))
            temp_seas = []
            for j in range(config['seasonality']):
                temp_seas.append(nn.Parameter(torch.Tensor([0.5])))
            init_seasonalities.append(torch.Tensor(nn.ParameterList(copy.copy(temp_seas))))

        self.init_lev_sms = nn.ParameterList(init_lev_sms)
        self.init_seas_sms = nn.ParameterList(init_seas_sms)
        self.init_seasonalities = init_seasonalities

        self.nl_layer = nn.Linear(config['state_hsize'], config['state_hsize'])
        self.act = nn.Tanh()
        self.scoring = nn.Linear(config['state_hsize'], config['output_size'])

        self.logistic = nn.Sigmoid()

    def forward(self, train, val, test, info_cat, idxs, add_nl_layer=False, testing=False):
        #         GET THE PER SERIES PARAMETERS
        lev_sms = self.logistic(torch.stack([self.init_lev_sms[idx] for idx in idxs]).squeeze(1))
        seas_sms = self.logistic(torch.stack([self.init_seas_sms[idx] for idx in idxs]).squeeze(1))
        init_seasonalities = torch.stack([self.init_seasonalities[idx] for idx in idxs])

        seasonalities = []
        # prime seasonality
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

        if self.config['level_variability_penalty'] > 0:
            sq_log_diff = torch.stack([(log_diff_of_levels[i] - log_diff_of_levels[i - 1]) ** 2 for i in range(1, len(log_diff_of_levels))])
            mean_sq_log_diff = torch.mean(sq_log_diff, dim=1)

        if self.config['output_size'] > self.config['seasonality']:
            start_seasonality_ext = seasonalities_stacked.shape[1] - self.config['seasonality']
            seasonalities_stacked = torch.cat((seasonalities_stacked, seasonalities_stacked[:, start_seasonality_ext:]), dim=1)



        window_input = []
        for i in range(self.config['input_size'] - 1, train.shape[1] - self.config['output_size']):
            input_window_start = i + 1 - self.config['input_size']
            input_window_end = i + 1

            train_deseas_window = train[:, input_window_start:input_window_end] / seasonalities_stacked[:, input_window_start:input_window_end]
            train_deseas_norm_window = (train_deseas_window / levs_stacked[:, i].unsqueeze(1))
            train_deseas_norm_cat_window = torch.cat((train_deseas_norm_window, info_cat), dim=1)
            window_input.append(train_deseas_norm_cat_window)

            output_window_start = i + 1
            output_window_end = i + 1 + self.config['output_size']

        input_batch = torch.cat([i.unsqueeze(0) for i in window_input], dim=0)

        print('done')

#         WINDOWING LOOP
#         TIME LOOP
#         RNN STUFF HERE

#         if add_nl_layer:
#             out = self.nl_layer(out)
#             out = self.act(out)
#         out = self.scoring(out)
