import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd



def read_file(file_location):
    series = []
    ids = []
    with open(file_location, 'r') as file:
        data = file.read().split("\n")

    for i in range(1, len(data) - 1):
        row = data[i].replace('"', '').split(',')
        series.append(np.array([float(j) for j in row[1:] if j != ""]))
        ids.append(row[0])

    series = np.array(series)
    return series


def create_val_set(train, output_size):
    val = []
    for i in range(len(train)):
        val.append(train[i][-output_size:])
        train[i] = train[i][:-output_size]
    return np.array(val)


def create_datasets(train_file_location, test_file_location, output_size):
    train = read_file(train_file_location)
    test = read_file(test_file_location)
    vals = create_val_set(train, output_size)
    return train, vals, test

class SeriesDataset(Dataset):

    def __init__(self, dataTrain, dataVal, dataTest, info, variable, device):
        self.dataInfoCatOHE = pd.get_dummies(info[info['SP'] == variable]['category'])
        self.dataInfoCatHeaders = self.dataInfoCatOHE.columns
        self.dataInfoCat = torch.from_numpy(self.dataInfoCatOHE.values)
        self.dataTrain = [torch.tensor(i) for i in dataTrain]
        self.dataVal = [torch.tensor(i) for i in dataVal]
        self.dataTest = [torch.tensor(i) for i in dataTest]
        self.device = device

    def __len__(self):
        return len(self.dataTrain)

    def __getitem__(self, idx):
        return self.dataTrain[idx].to(self.device), \
               self.dataVal[idx].to(self.device), \
               self.dataTest[idx].to(self.device), \
               self.dataInfoCat[idx].to(self.device), \
               idx


def collate_lines(seq_list):
    train_, val_, test_, info_cat_, idx_ = zip(*seq_list)
    train_lens = [len(seq) for seq in train_]
    seq_order = sorted(range(len(train_lens)), key=train_lens.__getitem__, reverse=True)
    train = [train_[i] for i in seq_order]
    val = [val_[i] for i in seq_order]
    test = [test_[i] for i in seq_order]
    info_cat = [info_cat_[i] for i in seq_order]
    idx = [idx_[i] for i in seq_order]
    return train, val, test, info_cat, idx

