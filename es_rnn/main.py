import pandas as pd
from torch.utils.data import DataLoader
from es_rnn.data_loading import create_datasets, SeriesDataset, collate_lines
from es_rnn.config import get_config
from es_rnn.trainer import ESRNNTrainer
from es_rnn.model import ESRNN
import time

print('loading config')
q_config = get_config()

print('loading data')
info = pd.read_csv('../data/info.csv')

if q_config['prod']:
    train_path = '../data/M4DataSetTrain/Quarterly-train.csv'
    test_path = '../data/M4DataSetTest/Quarterly-test.csv'
else:
    train_path = '../data/M4DataSetTrain/Quarterly-train-small.csv'
    test_path = '../data/M4DataSetTest/Quarterly-test-small.csv'

train, val, test = create_datasets(train_path,
                                   test_path,
                                   q_config['output_size'])

dataset = SeriesDataset(train, val, test, info, q_config['variable'], q_config['device'])
dataloader = DataLoader(dataset, batch_size=q_config['batch_size'], shuffle=True, collate_fn=collate_lines)

print('initializing model and train')
run_id = str(int(time.time()))
model = ESRNN(num_series=len(dataset), config=q_config)
tr = ESRNNTrainer(model, dataloader, run_id, q_config)
train, val, test, info_cat, idxs = iter(dataloader).next()
model.forward(train, val, test, info_cat, idxs)