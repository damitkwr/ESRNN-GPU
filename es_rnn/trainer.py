import os
import time
import numpy as np
import torch
import torch.nn as nn
from es_rnn.loss_modules import PinballLoss, sMAPE, np_sMAPE
from utils.logger import Logger
import pandas as pd


class ESRNNTrainer(nn.Module):
    def __init__(self, model, dataloader, run_id, config, ohe_headers):
        super(ESRNNTrainer, self).__init__()
        self.model = model.to(config['device'])
        self.grouped_results = None
        self.config = config
        self.dl = dataloader
        self.ohe_headers = ohe_headers
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'], eps=config['eps'])
        self.optimizer = torch.optim.ASGD(self.model.parameters(), lr=config['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=config['lr_anneal_step'],
                                                         gamma=config['lr_anneal_rate'])
        self.criterion = PinballLoss(self.config['training_tau'], self.config['output_size'], self.config['device'])
        self.epochs = 0
        self.max_epochs = config['num_of_train_epochs']
        self.run_id = str(run_id)
        self.prod_str = 'prod' if config['prod'] else 'dev'
        self.log = Logger("../logs/train%s%s%s" % (self.config['variable'], self.prod_str, self.run_id))

    def train_epochs(self):
        max_loss = 1e8
        for e in range(self.max_epochs):
            self.scheduler.step()
            epoch_loss = self.train()
            if epoch_loss < max_loss:
                self.save()
            if e % self.config['print_output_stats'] == 0:
                self.output_training_stats()

    def train(self):
        self.model.train()
        epoch_loss = 0
        for batch_num, (train, val, test, info_cat, idx) in enumerate(self.dl):
            start = time.time()
            print("Train_batch: %d" % (batch_num + 1))
            loss, hold_out_smape = self.train_batch(train, val, test, info_cat, idx)
            epoch_loss += loss
            end = time.time()
            self.log.log_scalar('Iteration time', end - start, batch_num + 1 * (self.epochs + 1))
        epoch_loss = epoch_loss / (batch_num + 1)
        self.epochs += 1

        # LOG EPOCH LEVEL INFORMATION
        print('[TRAIN]  Epoch [%d/%d]   Loss: %.4f, Hold Out sMAPE: %.4f' % (
            self.epochs, self.max_epochs, epoch_loss, hold_out_smape))
        info = {'loss': epoch_loss, 'hold_out_smape': hold_out_smape}
        self.log_values(info)

        return epoch_loss

    def train_batch(self, train, val, test, info_cat, idx):
        self.optimizer.zero_grad()
        network_pred, network_act, hold_out_pred, hold_out_act, loss_mean_sq_log_diff_level = self.model(train, val,
                                                                                                         test, info_cat,
                                                                                                         idx)

        hold_out_smape = sMAPE(hold_out_pred.view(-1), hold_out_act.view(-1), self.config['output_size'] * self.config[
            'batch_size'])

        loss = self.criterion(network_pred, network_act)
        loss = loss + loss_mean_sq_log_diff_level * self.config['level_variability_penalty']
        loss.backward()
        nn.utils.clip_grad_value_(self.model.parameters(), self.config['gradient_clipping'])
        self.optimizer.step()
        return float(loss), float(hold_out_smape)

    def output_training_stats(self):
        self.model.eval()
        acts = []
        preds = []
        info_cats = []
        for batch_num, (train, val, test, info_cat, idx) in enumerate(self.dl):
            _, _, hold_out_pred, hold_out_act, _ = self.model(train, val, test, info_cat, idx)

            acts.extend(hold_out_act.view(-1).cpu().detach().numpy())
            preds.extend(hold_out_pred.view(-1).cpu().detach().numpy())
            info_cats.append(info_cat.cpu().detach().numpy())
        info_cat_overall = np.concatenate(info_cats, axis=0)
        overall_hold_out_df = pd.DataFrame({'acts': acts, 'preds': preds})
        cats = [val for val in self.ohe_headers[info_cat_overall.argmax(axis=1)] for _ in
                range(self.config['output_size'])]
        overall_hold_out_df['category'] = cats
        self.grouped_results = overall_hold_out_df.groupby(['category']).apply(
            lambda x: np_sMAPE(x.preds, x.acts, x.shape[0]))

        file_path = os.path.join('..', 'grouped_results', self.run_id, self.prod_str)
        os.makedirs(file_path, exist_ok=True)

        print(self.grouped_results)
        grouped_path = os.path.join(file_path, 'grouped_results-{}.pyt'.format(self.epochs))
        self.grouped_results.to_csv(grouped_path)

    def save(self, save_dir='..'):
        print('Loss decreased, saving model!')
        file_path = os.path.join(save_dir, 'models', self.run_id, self.prod_str)
        model_path = os.path.join(file_path, 'model-{}.pyt'.format(self.epochs))
        os.makedirs(file_path, exist_ok=True)
        torch.save({'state_dict': self.model.state_dict()}, model_path)

    def log_values(self, info):

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
