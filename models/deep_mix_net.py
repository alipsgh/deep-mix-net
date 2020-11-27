
import numpy as np
import torch

from torch.nn import Module
from sklearn.metrics import accuracy_score
from torch import nn, optim
from torch.optim.lr_scheduler import MultiplicativeLR


class DeepSeqNet(Module):

    def __init__(self):
        super(DeepSeqNet, self).__init__()

    def _compile(self, optimizer, learning_rate):
        self._set_optim(optimizer, learning_rate)
        self._set_scheduler()
        self._set_criterion()

    def _set_optim(self, optimizer, learning_rate):
        optimizer = optimizer.lower()
        if optimizer == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)

    def _set_scheduler(self):
        self.scheduler = MultiplicativeLR(self.optimizer, lr_lambda=(lambda x: 0.95))

    def _set_criterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x_txt, x_num):

        txt_features = self.txt_net_forward(x_txt)
        num_features = self.num_net_forward(x_num)

        features = torch.cat((txt_features, num_features), 1)
        out_features = self.dropout(features)

        logits = self.fc(out_features)

        return logits

    def txt_net_forward(self, x_txt):
        raise NotImplementedError()

    def num_net_forward(self, x_num):
        for linear in self.linear_layers:
            x_num = self.activation_layer(linear(x_num))
        return x_num

    def fit(self, x_txt, x_num, y):

        self.train()

        self.optimizer.zero_grad()

        y_ = self.forward(x_txt, x_num)

        loss = self.criterion(y_, y)
        loss.backward()

        self.optimizer.step()

        return loss

    def evaluate(self, data_iterator):

        self.eval()

        labels, preds = [], []
        for _, (x_txt, x_num, y) in enumerate(data_iterator):

            x_txt, x_num = x_txt.t(), x_num.t()
            if torch.cuda.is_available():
                x_txt, x_num = x_txt.cuda(), x_num.cuda()

            y_ = self.forward(x_txt, x_num)
            pred = torch.argmax(y_, 1)

            preds.extend(pred.cpu().numpy())
            labels.extend(y.numpy())

        score = accuracy_score(labels, np.array(preds).flatten())

        return score

    def run_epoch(self, train_iterator, val_iterator):

        train_losses = []
        val_accuracies = []
        losses = []
        for i, (x_txt, x_num, y) in enumerate(train_iterator):

            x_txt, x_num = x_txt.t(), x_num.t()
            if torch.cuda.is_available():
                x_txt, x_tab = x_txt.cuda(), x_num.cuda()
                y = y.cuda()

            loss = self.fit(x_txt, x_num, y)
            losses.append(loss.item())

            if i % 100 == 0 and i != 0:
                avg_train_loss = float(np.mean(losses))
                train_losses.append(avg_train_loss)
                losses = []

                val_accuracy = self.evaluate(val_iterator)
                print("Iteration: %4d | train loss: %3.2f | val acc.: %.2f" % ((i + 1),
                                                                               avg_train_loss * 100,
                                                                               val_accuracy * 100))

        # Run the scheduler to reduce the learning rate
        self.scheduler.step(epoch=None)

        return train_losses, val_accuracies

