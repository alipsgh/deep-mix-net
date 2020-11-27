
import torch

from models.deep_mix_net import DeepSeqNet
from torch import nn
from torch.nn import functional as F


class AttentionRNN(DeepSeqNet):

    def __init__(self, vocab_size, embeddings, embedding_size, lstm_hidden_dim, lstm_num_hidden_layers, bidirectional,
                 tab_input_dim, linear_layers_dim, output_dim, dropout_rate, optimizer="adam", learning_rate=0.01):

        super(AttentionRNN, self).__init__()

        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        # =======================
        #  Attention RNN Network
        # =======================
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_hidden_layers = lstm_num_hidden_layers
        self.bidirectional = bidirectional

        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embeddings.weight = nn.Parameter(embeddings, requires_grad=False)

        self.lstm = nn.LSTM(input_size=self.embedding_size,
                            hidden_size=self.lstm_hidden_dim,
                            num_layers=self.lstm_num_hidden_layers,
                            bidirectional=self.bidirectional)

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
        self.fc = nn.Linear(self.lstm_hidden_dim * (1 + self.bidirectional) * 2 + self.linear_layers[-1].out_features,
                            self.output_dim)
        self.softmax = nn.Softmax(dim=1)

        self.optimizer, self.scheduler, self.criterion = None, None, None
        self._compile(optimizer, learning_rate)

        if torch.cuda.is_available():
            self.cuda()

    @staticmethod
    def apply_attention(rnn_output, final_hidden_state):
        """
        Apply Attention on RNN output
        :param rnn_output: (batch_size, seq_len, num_directions * hidden_size): tensor representing hidden state for every word in the sentence
        :param final_hidden_state: (batch_size, num_directions * hidden_size): final hidden state of the RNN
        :return:
        """
        hidden_state = final_hidden_state.unsqueeze(2)
        attention_scores = torch.bmm(rnn_output, hidden_state).squeeze(2)
        soft_attention_weights = F.softmax(attention_scores, 1).unsqueeze(2)  # >> shape = (batch_size, seq_len, 1)
        attention_output = torch.bmm(rnn_output.permute(0, 2, 1), soft_attention_weights).squeeze(2)
        return attention_output

    def txt_net_forward(self, x_txt):

        embedded_sequence = self.embeddings(x_txt)
        embedded_sequence = embedded_sequence.permute(1, 0, 2)
        # >> embedded_sequence: (seq_len, batch_size, embedding_size)

        o_n, (h_n, c_n) = self.lstm(embedded_sequence)
        # >> o_n: (seq_len, batch_size, num_directions * hidden_size)
        # >> h_n: (num_directions, batch_size, hidden_size)

        batch_size = h_n.shape[1]
        final_h_n = h_n.view(self.lstm_num_hidden_layers, self.bidirectional + 1, batch_size,
                             self.lstm_hidden_dim)[-1, :, :, :]

        final_hidden_state = torch.cat([final_h_n[i, :, :] for i in range(final_h_n.shape[0])], dim=1)

        attention_out = self.apply_attention(o_n.permute(1, 0, 2), final_hidden_state)
        # >> attention_out: (batch_size, num_directions * hidden_size)

        feature_vector = torch.cat([final_hidden_state, attention_out], dim=1)

        return feature_vector

