from math import sqrt

import torch


def get_config():
    q_config = {
        #     RUNTIME PARAMETERS
        'prod': False,
        'device': ("cuda" if torch.cuda.is_available() else "cpu"),
        'lback': False,
        'chop_val': 72,

        #     MODEL STRUCTURE PARAMETER

        'variable': "Quarterly",
        'run': "50/45 (1,2),(4,8), LR=0.001/{10,1e-4f}, EPOCHS=15, LVP=80 40*",
        'percentile': 50,
        'training_percentile': 45,
        'dilations': ((1, 2), (4, 8)),
        'use_residual_lstm': False,
        'add_nl_layer': False,
        'learning_rate': 1e-3,
        'learning_rates': ((10, 1e-4)),
        'per_series_lr_multip': 1,
        'num_of_train_epochs': 15,
        'state_hsize': 40,
        'seasonality': 4,
        'input_size': 4,
        'output_size': 8,
        'min_inp_seq_len': 0,
        'level_variability_penalty': 80,
        'batch_size': 128,
        'num_of_categories': 6,  # in data provided
        'big_loop': 3,
        'num_of_chunks': 2,
        'eps': 1e-6,
        'averaging_level': 5,
        'use_median': False,
        'middle_pos_for_avg': 2,  # if using medians
        'noise_std': 0.001,
        'freq_of_test': 1,
        'gradient_clipping': 20,
        'c_state_penalty': 0,
        'big_float': 1e38,  # numeric_limits<float>::max(),
        'print_diagn': True,
        'max_num_of_series': -1,

        'use_auto_learning_rate': False,
        'min_learning_rate': 0.0001,
        'lr_ratio': sqrt(10),
        'lr_tolerance_multip': 1.005,
        'l3_period': 2,
        'min_epochs_before_changing_lrate': 2,
        'print_train_batch_every': 5,
        'lr_anneal_rate': 0.5,
        'lr_anneal_step': 5

    }

    q_config['input_size_i'] = q_config['input_size']
    q_config['output_size_i'] = q_config['output_size']
    q_config['min_series_length'] = q_config['input_size_i'] + q_config['output_size_i'] + q_config[
        'min_inp_seq_len'] + 2
    q_config['max_series_length'] = 40 * q_config['seasonality'] + q_config['min_series_length']
    q_config['tau'] = q_config['percentile'] / 100
    q_config['training_tau'] = q_config['training_percentile'] / 100
    q_config['attention_hsize'] = q_config['state_hsize']

    if not q_config['prod']:
        q_config['batch_size'] = 10
        q_config['max_num_of_series'] = 40

    return q_config
