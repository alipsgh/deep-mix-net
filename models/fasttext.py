
import torch

from models.deep_mix_net import DeepSeqNet
from torch import nn


class FastText(DeepSeqNet):

    def __init__(self, vocab_size, embeddings, embedding_size, hidden_layer_size,
                 tab_input_dim, linear_layers_dim, output_dim, dropout_rate, optimizer="adam", learning_rate=0.01):

        super(FastText, self).__init__()

        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        # ==========
        #  FastText
        # ==========

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_layer_size = hidden_layer_size

        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embeddings.weight = nn.Parameter(embeddings, requires_grad=False)
        self.embed_fc_layer = nn.Linear(self.embedding_size, self.hidden_layer_size)

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
        self.fc = nn.Linear(self.hidden_layer_size + self.linear_layers[-1].out_features, self.output_dim)
        self.softmax = nn.Softmax(dim=1)

        self.optimizer, self.scheduler, self.criterion = None, None, None
        self._compile(optimizer, learning_rate)

        if torch.cuda.is_available():
            self.cuda()

    def txt_net_forward(self, x_txt):
        embedded_sequence = self.embeddings(x_txt)
        feature_vector = self.embed_fc_layer(embedded_sequence.mean(1))
        return feature_vector

