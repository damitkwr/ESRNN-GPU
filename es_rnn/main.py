import pandas as pd
from torch.utils.data import DataLoader
from es_rnn.data_loading import create_datasets, SeriesDataset
from es_rnn.config import get_config
from es_rnn.trainer import ESRNNTrainer
from es_rnn.model import ESRNN
import numpy as np
import time
import random
import torch

print('loading config')
config = get_config('Quarterly')

if not config['prod']:
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

print('loading data')
dev_flag = '-small' if not config['prod'] else ''

info = pd.read_csv('../data/info%s.csv' % (dev_flag))
cat_idx = pd.Index(pd.unique(info['category']))

info['category'] = pd.Categorical(cat_idx.get_indexer(info['category']), cat_idx)
train_path = '../data/Train/%s-train%s.csv' % (config['variable'], dev_flag)
test_path = '../data/Test/%s-test%s.csv' % (config['variable'], dev_flag)

train, val, test = create_datasets(train_path, test_path, config['output_size'])

dataset = SeriesDataset(train, val, test, info, config['variable'], config['chop_val'], config['device'])
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=config['prod'])

run_id = str(int(time.time()))
model = ESRNN(num_series=len(dataset), config=config)
tr = ESRNNTrainer(model, dataloader, run_id, config, ohe_headers=dataset.dataInfoCatHeaders)
tr.train_epochs()
