
import torch

from models.deep_mix_net import DeepSeqNet
from torch import nn


class TextRNN(DeepSeqNet):

    def __init__(self, vocab_size, embeddings, embedding_size, rnn_hidden_size, rnn_num_hidden_layers, rnn_bidirectional,
                 tab_input_dim, linear_layers_dim, output_dim, dropout_rate, optimizer, learning_rate):

        super(TextRNN, self).__init__()

        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        # =========
        #  TextRNN
        # =========

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_hidden_layers = rnn_num_hidden_layers
        self.rnn_bidirectional = rnn_bidirectional

        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embeddings.weight = nn.Parameter(embeddings, requires_grad=False)

        self.lstm = nn.LSTM(input_size=self.embedding_size,
                            hidden_size=self.rnn_hidden_size,
                            num_layers=self.rnn_num_hidden_layers,
                            dropout=self.dropout_rate,
                            bidirectional=self.rnn_bidirectional)

        # ==============================
        #  Feed Forward Neural Networks
        # ==============================
        self.linear_layers = nn.ModuleList()
        self.activation_layer = nn.ReLU()

        for i, hidden_dim in enumerate(linear_layers_dim):
            if i == 0:
                self.linear_layers.append(nn.Linear(tab_input_dim, hidden_dim))
            else:
                self.linear_layers.append(nn.Linear(self.linear_layers[-1].out_features, hidden_dim))

        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc = nn.Linear(self.rnn_hidden_size * self.rnn_num_hidden_layers * (1 + self.rnn_bidirectional) + self.linear_layers[-1].out_features, self.output_dim)

        self.softmax = nn.Softmax(dim=1)

        self.optimizer, self.scheduler, self.criterion = None, None, None
        self._compile(optimizer, learning_rate)

        if torch.cuda.is_available():
            self.cuda()

    def txt_net_forward(self, x_txt):

        # >> x: (max_sen_len, batch_size)
        embedded_sequence = self.embeddings(x_txt)
        # >> embedded_sequence: (seq_len, batch_size, embedding_size)
        embedded_sequence = embedded_sequence.permute(1, 0, 2)
        # >> h_n: (num_layers * num_directions, batch_size, hidden_size)
        o_n, (h_n, c_n) = self.lstm(embedded_sequence)
        feature_vec = self.dropout(h_n)
        # >> feature_vec: (batch_size, hidden_size * hidden_layers * num_directions) > reshaping is for the linear layer
        feature_vec = torch.cat([feature_vec[i, :, :] for i in range(feature_vec.shape[0])], dim=1)

        return feature_vec

