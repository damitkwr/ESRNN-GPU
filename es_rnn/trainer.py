import os
import time
import numpy as np
import copy
import torch
import torch.nn as nn
from es_rnn.loss_modules import PinballLoss, sMAPE, np_sMAPE
from utils.logger import Logger
import pandas as pd


class ESRNNTrainer(nn.Module):
    def __init__(self, model, dataloader, run_id, config, ohe_headers):
        super(ESRNNTrainer, self).__init__()
        self.model = model.to(config['device'])
        self.config = config
        self.dl = dataloader
        self.ohe_headers = ohe_headers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        # self.optimizer = torch.optim.ASGD(self.model.parameters(), lr=config['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=config['lr_anneal_step'],
                                                         gamma=config['lr_anneal_rate'])
        self.criterion = PinballLoss(self.config['training_tau'],
                                     self.config['output_size'] * self.config['batch_size'], self.config['device'])
        self.epochs = 0
        self.max_epochs = config['num_of_train_epochs']
        self.run_id = str(run_id)
        self.prod_str = 'prod' if config['prod'] else 'dev'
        self.log = Logger("../logs/train%s%s%s" % (self.config['variable'], self.prod_str, self.run_id))
        self.csv_save_path = None

    def train_epochs(self):
        max_loss = 1e8
        start_time = time.time()
        for e in range(self.max_epochs):
            self.scheduler.step()
            epoch_loss = self.train()
            if epoch_loss < max_loss:
                self.save()
            epoch_val_loss = self.val()
            if e == 0:
                file_path = os.path.join(self.csv_save_path, 'validation_losses.csv')
                with open(file_path, 'w') as f:
                    f.write('epoch,training_loss,validation_loss\n')
            with open(file_path, 'a') as f:
                f.write(','.join([str(e), str(epoch_loss), str(epoch_val_loss)]) + '\n')
        print('Total Training Mins: %5.2f' % ((time.time()-start_time)/60))

    def train(self):
        self.model.train()
        epoch_loss = 0
        for batch_num, (train, val, test, info_cat, idx) in enumerate(self.dl):
            start = time.time()
            print("Train_batch: %d" % (batch_num + 1))
            loss = self.train_batch(train, val, test, info_cat, idx)
            epoch_loss += loss
            end = time.time()
            self.log.log_scalar('Iteration time', end - start, batch_num + 1 * (self.epochs + 1))
        epoch_loss = epoch_loss / (batch_num + 1)
        self.epochs += 1

        # LOG EPOCH LEVEL INFORMATION
        print('[TRAIN]  Epoch [%d/%d]   Loss: %.4f' % (
            self.epochs, self.max_epochs, epoch_loss))
        info = {'loss': epoch_loss}

        self.log_values(info)
        self.log_hists()

        return epoch_loss

    def train_batch(self, train, val, test, info_cat, idx):
        self.optimizer.zero_grad()
        network_pred, network_act, _, _, loss_mean_sq_log_diff_level = self.model(train, val,
                                                                                  test, info_cat,
                                                                                  idx)

        loss = self.criterion(network_pred, network_act)
        loss.backward()
        nn.utils.clip_grad_value_(self.model.parameters(), self.config['gradient_clipping'])
        self.optimizer.step()
        return float(loss)

    def val(self):
        self.model.eval()
        with torch.no_grad():
            acts = []
            preds = []
            info_cats = []

            hold_out_loss = 0
            for batch_num, (train, val, test, info_cat, idx) in enumerate(self.dl):
                _, _, (hold_out_pred, network_output_non_train), \
                (hold_out_act, hold_out_act_deseas_norm), _ = self.model(train, val, test, info_cat, idx)
                hold_out_loss += self.criterion(network_output_non_train.unsqueeze(0).float(),
                                                hold_out_act_deseas_norm.unsqueeze(0).float())
                acts.extend(hold_out_act.view(-1).cpu().detach().numpy())
                preds.extend(hold_out_pred.view(-1).cpu().detach().numpy())
                info_cats.append(info_cat.cpu().detach().numpy())
            hold_out_loss = hold_out_loss / (batch_num + 1)

            info_cat_overall = np.concatenate(info_cats, axis=0)
            _hold_out_df = pd.DataFrame({'acts': acts, 'preds': preds})
            cats = [val for val in self.ohe_headers[info_cat_overall.argmax(axis=1)] for _ in
                    range(self.config['output_size'])]
            _hold_out_df['category'] = cats

            overall_hold_out_df = copy.copy(_hold_out_df)
            overall_hold_out_df['category'] = ['Overall' for _ in cats]

            overall_hold_out_df = pd.concat((_hold_out_df, overall_hold_out_df))
            grouped_results = overall_hold_out_df.groupby(['category']).apply(
                lambda x: np_sMAPE(x.preds, x.acts, x.shape[0]))

            results = grouped_results.to_dict()
            results['hold_out_loss'] = float(hold_out_loss.detach().cpu())

            self.log_values(results)

            file_path = os.path.join('..', 'grouped_results', self.run_id, self.prod_str)
            os.makedirs(file_path, exist_ok=True)

            print(results)
            grouped_path = os.path.join(file_path, 'grouped_results-{}.csv'.format(self.epochs))
            grouped_results.to_csv(grouped_path)
            self.csv_save_path = file_path

        return hold_out_loss.detach().cpu().item()

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

    def log_hists(self):
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
