
import torch

from models.deep_mix_net import DeepSeqNet
from torch import nn
from torch.nn import functional as F


class RCNN(DeepSeqNet):

    def __init__(self, vocab_size, embeddings, embedding_size,
                 rcnn_num_hidden_layers, rcnn_hidden_size, rcnn_linear_size,
                 output_dim, dropout_rate, linear_layers_dim, tab_input_dim,
                 optimizer, learning_rate):

        super(RCNN, self).__init__()

        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        # =========
        #   R-CNN
        # =========
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.rcnn_num_hidden_layers = rcnn_num_hidden_layers
        self.rcnn_hidden_size = rcnn_hidden_size
        self.rcnn_linear_size = rcnn_linear_size
        
        # Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, self.embedding_size)
        self.embeddings.weight = nn.Parameter(embeddings, requires_grad=False)

        # BiLSTM
        self.lstm = nn.LSTM(input_size=self.embedding_size,
                            num_layers=self.rcnn_num_hidden_layers,
                            hidden_size=self.rcnn_hidden_size,
                            dropout=self.dropout_rate,
                            bidirectional=True)
        
        # Linear layer to get "convolution output" to be passed to Pooling Layer
        self.linear = nn.Sequential(nn.Linear(self.embedding_size + 2 * self.rcnn_hidden_size,
                                              self.rcnn_linear_size), nn.Tanh())
        self.dropout = nn.Dropout(self.dropout_rate)

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

        self.fc = nn.Linear(self.rcnn_linear_size + self.linear_layers[-1].out_features, self.output_dim)
        self.softmax = nn.Softmax(dim=1)

        self.optimizer, self.scheduler, self.criterion = None, None, None
        self._compile(optimizer, learning_rate)

        if torch.cuda.is_available():
            self.cuda()

    def txt_net_forward(self, x_txt):

        embedded_sequence = self.embeddings(x_txt)
        # >> embedded_sequence: (seq_len, batch_size, embed_size)
        embedded_sequence = embedded_sequence.permute(1, 0, 2)

        # >> o_n: (seq_len, batch_size, 2 * hidden_size)
        o_n, (_, _) = self.lstm(embedded_sequence)
        # >> input_features: (batch_size, seq_len, embed_size + 2 * hidden_size)
        input_features = torch.cat([o_n, embedded_sequence], 2).permute(1, 0, 2)

        # >> linear_output: (batch_size, seq_len, hidden_size_linear)
        linear_output = self.linear(input_features)
        # >> Reshaping for max_pool
        linear_output = linear_output.permute(0, 2, 1)

        # >> out_features: (batch_size, hidden_size_linear)
        out_features = F.max_pool1d(linear_output, linear_output.shape[2]).squeeze(2)

        return out_features

