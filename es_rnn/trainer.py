import torch.nn as nn
import torch
from utils.logger import Logger

# THIS IS JUST A START THIS CLASS NEEDS TO CHANGE A TON!


class ESRNNTrainer(nn.Module):
    def __init__(self, model, dataloader, run_id, config):
        super(ESRNNTrainer, self).__init__()
        self.model = model.to(config['device'])
        self.dl = dataloader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'], eps=config['eps'])
        self.criterion = None
        self.epochs = 0
        self.max_epochs = config['num_of_train_epochs']
        self.run_id = str(run_id)
        self.prod_str = 'prod' if config['prod'] else 'dev'
        self.log = Logger("../logs/train%s%s" % (self.prod_str, self.run_id))

    def train(self):
        self.model.train()
        epoch_loss = 0
        num_batches = 0
        for batch_num, (train, val, test, info_cat, idx) in enumerate(self.dl):
            if batch_num % config['print_train_batch_every'] == 0:
                print("train_batch: %d" % batch_num)
            loss = self.train_batch(train, val, test, info_cat, idx)
            epoch_loss += loss
        epoch_loss = epoch_loss / (batch_num + 1)
        self.epochs += 1
        print('[TRAIN]  Epoch [%d/%d]   Loss: %.4f'
              % (self.epochs, self.max_epochs, epoch_loss))
        info = {'loss': epoch_loss}
        for tag, value in info.items():
            self.log.log_scalar(tag, value, self.epochs + 1)
        return epoch_loss

    def train_batch(self, train, val, test, info_cat, idx):
        self.optimizer.zero_grad()
        output = self.model(train, val, test, info_cat, idx)
        loss = self.criterion(output, )
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clipping'])
        self.optimizer.step()
        return float(loss)
