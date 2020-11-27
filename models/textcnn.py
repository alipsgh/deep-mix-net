
import torch

from models.deep_mix_net import DeepSeqNet
from torch import nn


class TextCNN(DeepSeqNet):

    def __init__(self, vocab_size, embeddings, embedding_size, num_channels, kernel_size, max_seq_len,
                 tab_input_dim, linear_layers_dim, output_dim, dropout_rate, optimizer="adam", learning_rate=0.01):

        super(TextCNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.max_seq_len = max_seq_len
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        # =================
        #  TextCNN Network
        # =================
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embeddings.weight = nn.Parameter(embeddings, requires_grad=False)

        self.conv_1 = nn.Sequential(nn.Conv1d(in_channels=self.embedding_size, out_channels=self.num_channels,
                                              kernel_size=self.kernel_size[0]), nn.ReLU())
        self.conv_2 = nn.Sequential(nn.Conv1d(in_channels=self.embedding_size, out_channels=self.num_channels,
                                              kernel_size=self.kernel_size[1]), nn.ReLU())
        self.conv_3 = nn.Sequential(nn.Conv1d(in_channels=self.embedding_size, out_channels=self.num_channels,
                                              kernel_size=self.kernel_size[2]), nn.ReLU())

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
        self.fc = nn.Linear(in_features=self.num_channels * len(self.kernel_size) + self.linear_layers[-1].out_features,
                            out_features=self.output_dim)
        self.softmax = nn.Softmax(dim=1)

        self.optimizer, self.scheduler, self.criterion = None, None, None
        self._compile(optimizer, learning_rate)

        if torch.cuda.is_available():
            self.cuda()

    def txt_net_forward(self, x_txt):

        embedded_sequence = self.embeddings(x_txt)
        embedded_sequence = embedded_sequence.permute(0, 2, 1)

        feature_map_1 = torch.max(self.conv_1(embedded_sequence), dim=2)[0]
        feature_map_2 = torch.max(self.conv_2(embedded_sequence), dim=2)[0]
        feature_map_3 = torch.max(self.conv_3(embedded_sequence), dim=2)[0]
        feature_map = torch.cat((feature_map_1, feature_map_2, feature_map_3), 1)

        return feature_map

