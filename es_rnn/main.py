import pandas as pd
from torch.utils.data import DataLoader
from es_rnn.data_loading import create_datasets, SeriesDataset, collate_lines
from es_rnn.config import get_config
from es_rnn.trainer import ESRNNTrainer
from es_rnn.model import ESRNN
import time

q_config = get_config()
info = pd.read_csv('../data/info.csv')
train, val, test = create_datasets('../data/M4DataSetTrain/Quarterly-train.csv',
                                   '../data/M4DataSetTest/Quarterly-test.csv',
                                   q_config['output_size'])

dataset = SeriesDataset(train, val, test, info, q_config['variable'], q_config['device'])
dataloader = DataLoader(dataset, batch_size=q_config['batch_size'], shuffle=True, collate_fn=collate_lines)

run_id = str(int(time.time()))
model = ESRNN(num_series=len(dataset), config=q_config)
tr = ESRNNTrainer(model, dataloader, run_id, q_config)
