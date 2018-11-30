import torch
import torch.nn as nn


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
            init_seasonalities.append(nn.Parameter(torch.ones(config['seasonality']) * 0.5))

        self.init_lev_sms = nn.ParameterList(init_lev_sms)
        self.init_seas_sms = nn.ParameterList(init_seas_sms)
        self.init_seasonalities = nn.ParameterList(init_seasonalities)

        self.nl_layer = nn.Linear(config['state_hsize'], config['state_hsize'])
        self.act = nn.Tanh()
        self.scoring = nn.Linear(config['state_hsize'], config['output_size'])

    def forward(self, train, val, test, info_cat, idxs, add_nl_layer):
        #         GET THE PER SERIES PARAMETERS
        lev_sms = torch.stack([self.init_lev_sms[idx] for idx in idxs])
        seas_sms = torch.stack([self.init_seas_sms[idx] for idx in idxs])
        seasonalities = torch.stack([self.init_seasonalities[idx] for idx in idxs])

#         WINDOWING LOOP
#         TIME LOOP
#         RNN STUFF HERE

#         if add_nl_layer:
#             out = self.nl_layer(out)
#             out = self.act(out)
#         out = self.scoring(out)
