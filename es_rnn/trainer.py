import numpy as np
import os
import torch.nn as nn
import torch
from utils.logger import Logger
from es_rnn.loss_modules import PinballLoss


class ESRNNTrainer(nn.Module):
    def __init__(self, model, dataloader, run_id, config):
        super(ESRNNTrainer, self).__init__()
        self.model = model.to(config['device'])
        self.config = config
        self.dl = dataloader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'], eps=config['eps'])
        self.criterion = PinballLoss(self.config['training_tau'], self.config['output_size'], self.config['device'])
        self.epochs = 0
        self.max_epochs = config['num_of_train_epochs']
        self.run_id = str(run_id)
        self.prod_str = 'prod' if config['prod'] else 'dev'
        self.log = Logger("../logs/train%s%s" % (self.prod_str, self.run_id))

    def train(self):
        self.model.train()
        epoch_loss = 0
        for i in range(self.max_epochs):
            for batch_num, (train, val, test, info_cat, idx) in enumerate(self.dl):
                print("Train_batch: %d" % (batch_num + 1))
                train, val, test = train.to(self.config['device']), val.to(self.config['device']), test.to(
                    self.config['device'])

                loss = self.train_batch(train, val, test, info_cat, idx)
                epoch_loss += loss
            epoch_loss = epoch_loss / (batch_num + 1)
            self.epochs += 1

            print('[TRAIN]  Epoch [%d/%d]   Loss: %.4f' % (self.epochs, self.max_epochs, epoch_loss))
            info = {'loss': epoch_loss}
            self.log_hist_values(info)

            return epoch_loss

    def train_batch(self, train, val, test, info_cat, idx):
        self.optimizer.zero_grad()
        out, out_batch = self.model(train, val, test, info_cat, idx)

        loss = self.criterion(out, out_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clipping'])
        self.optimizer.step()
        return float(loss)

    def save(self, save_dir='.'):
        file_path = os.path.join(save_dir, 'models', self.run_id, self.prod_str)
        model_path = os.path.join(save_dir, 'models', self.run_id, self.prod_str, 'model-{}.pyt'.format(self.epochs))
        os.makedirs(file_path, exist_ok=True)
        torch.save({'state_dict': self.model.state_dict()}, model_path)

    def log_hist_values(self, info):

        # SCALAR
        for tag, value in info.items():
            self.log.log_scalar(tag, value, self.epochs + 1)

        # HISTS
        batch_params = dict()
        for tag, value in self.model.named_parameters():
            if value.grad is not None:
                if "init" in tag:
                    name, _ = tag.split(".")
                    if name not in batch_params.keys() or "%s/grad" % name not in batch_params.keys():
                        batch_params[name] = []
                        batch_params["%s/grad" % name] = []
                    batch_params[name].append(value.data.cpu().numpy())
                    batch_params["%s/grad" % name].append(value.grad.cpu().numpy())
                else:
                    tag = tag.replace('.', '/')
                    self.log.log_histogram(tag, value.data.cpu().numpy(), self.epochs + 1)
                    self.log.log_histogram(tag + '/grad', value.grad.data.cpu().numpy(), self.epochs + 1)
            else:
                print('Not printing %s because it\'s not updating' % tag)

        for tag, v in batch_params.items():
            vals = np.concatenate(np.array(v))
            self.log.log_histogram(tag, vals, self.epochs + 1)
