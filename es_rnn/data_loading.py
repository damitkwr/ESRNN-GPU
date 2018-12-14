import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


def read_file(file_location):
    series = []
    ids = []
    with open(file_location, 'r') as file:
        data = file.read().split("\n")

    for i in range(1, len(data) - 1):
        # for i in range(1, len(data)):
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


def chop_series(train, chop_val):
    # CREATE MASK FOR VALUES TO BE CHOPPED
    train_len_mask = [True if len(i) >= chop_val else False for i in train]
    # FILTER AND CHOP TRAIN
    train = [train[i][-chop_val:] for i in range(len(train)) if train_len_mask[i]]
    return train, train_len_mask


def create_datasets(train_file_location, test_file_location, output_size):
    train = read_file(train_file_location)
    test = read_file(test_file_location)
    val = create_val_set(train, output_size)
    return train, val, test


class SeriesDataset(Dataset):

    def __init__(self, dataTrain, dataVal, dataTest, info, variable, chop_value, device):
        dataTrain, mask = chop_series(dataTrain, chop_value)

        self.dataInfoCatOHE = pd.get_dummies(info[info['SP'] == variable]['category'])
        self.dataInfoCatHeaders = np.array([i for i in self.dataInfoCatOHE.columns.values])
        self.dataInfoCat = torch.from_numpy(self.dataInfoCatOHE[mask].values).float()
        self.dataTrain = [torch.tensor(dataTrain[i]) for i in range(len(dataTrain))]  # ALREADY MASKED IN CHOP FUNCTION
        self.dataVal = [torch.tensor(dataVal[i]) for i in range(len(dataVal)) if mask[i]]
        self.dataTest = [torch.tensor(dataTest[i]) for i in range(len(dataTest)) if mask[i]]
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

