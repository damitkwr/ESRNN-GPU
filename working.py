from torch.utils.data import Dataset, DataLoader


q_config = {
    'variable': "Quarterly",
    'run': "50/45 (1,2),(4,8), LR=0.001/{10,1e-4f}, EPOCHS=15, LVP=80 40*",
    'percentile': 50,
    'training_percentile': 45,
    'dilations': ((1, 2), (4, 8)),
    'use_residual_lstm': False,
    'add_nl_layer': False,
    'initial_learning_rate': 1e-3,
    'learning_rates': ((10, 1e-4)),
    'per_series_lr_multip': 1,
    'num_of_train_epochs': 15,
    'state_hsize': 40,
    'seasonality': 4,
    'input_size': 4,
    'output_size': 8,
    'min_inp_seq_len': 0,
    'level_variability_penalty': 80
}
q_config['input_size_i'] = q_config['input_size']
q_config['output_size_i'] = q_config['output_size']
q_config['min_series_length'] = q_config['input_size_i'] + q_config['output_size_i'] + q_config['min_inp_seq_len'] + 2
q_config['max_series_length'] = 40 * q_config['seasonality'] + q_config['min_series_length']

class SeriesDataset(Dataset):
    def __


