
import pandas as pd
import spacy
import torch

from torchtext import data
from torchtext.vocab import Vectors

from utils.logger import get_logger


class BatchGenerator:

    def __init__(self, data_iterator, txt_col, num_cols, trg_col):
        self.data_iterator = data_iterator
        self.txt_col = txt_col
        self.num_cols = num_cols
        self.trg_col = trg_col

    def __len__(self):
        return len(self.data_iterator)

    def __iter__(self):
        for batch in self.data_iterator:
            txt_data = getattr(batch, self.txt_col)
            num_data = torch.stack([getattr(batch, _) for _ in self.num_cols])
            y = getattr(batch, self.trg_col)
            yield txt_data, num_data, y


class Dataset(object):

    def __init__(self, data_columns, batch_size, seq_len=None):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab = None
        self.embeddings = None
        self.train_batch_loader = None
        self.valid_batch_loader = None
        self.test_batch_loader = None
        self.nlp = spacy.load('en_core_web_sm')
        self.data_cols = data_columns

    def tokenize(self, sent):
        output = [x.text for x in self.nlp.tokenizer(sent) if x.text != " "]
        return output

    def load_data(self, train_file, test_file, embedding_source=None):
        """
        Loads the data from files
        Sets up iterators for training, validation and test data
        Also create vocabulary and word embeddings based on the data
        :param train_file: absolute path to the training file
        :param test_file: absolute path to the test file
        :param embedding_source: absolute path to file containing word embeddings (GloVe/Word2Vec)
        :return:
        """

        # Let's load training and test data
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)

        # Creating data fields for data
        # If the self.seq_len is none, then the length will be flexible.
        text = data.Field(sequential=True, tokenize=self.tokenize, lower=True, fix_length=self.seq_len)
        label = data.Field(sequential=False, use_vocab=False, is_target=True)

        data_fields = []
        for k, v in self.data_cols.items():
            if k == "txt_col":
                data_fields.append(("text", text))
            elif k == "trg_col":
                data_fields.append(("label", label))
            else:
                for num_col in v:
                    data_fields.append((num_col, data.Field(sequential=False, dtype=torch.float32, use_vocab=False)))

        # Create torch datasets
        # 1. Training data
        train_examples = [data.Example.fromlist(example, data_fields) for example in train_df.values.tolist()]
        train_data = data.Dataset(train_examples, data_fields)
        # 2. Test data
        test_examples = [data.Example.fromlist(example, data_fields) for example in test_df.values.tolist()]
        test_data = data.Dataset(test_examples, data_fields)
        # 3. Validation data
        test_data, valid_data = test_data.split(split_ratio=0.5)

        if embedding_source is None:
            text.build_vocab(train_data)
        else:
            text.build_vocab(train_data, vectors=Vectors(embedding_source))
            self.embeddings = text.vocab.vectors

        self.vocab = text.vocab

        train_iterator = data.BucketIterator(train_data, batch_size=self.batch_size, sort_key=lambda x: len(x.text), repeat=False, shuffle=True)
        valid_iterator = data.BucketIterator(valid_data, batch_size=self.batch_size, sort_key=lambda x: len(x.text), repeat=False, shuffle=False)
        test_iterator = data.BucketIterator(test_data, batch_size=self.batch_size, sort_key=lambda x: len(x.text), repeat=False, shuffle=False)

        self.train_batch_loader = BatchGenerator(train_iterator, txt_col="text", num_cols=self.data_cols["num_col"], trg_col="label")
        self.valid_batch_loader = BatchGenerator(valid_iterator, txt_col="text", num_cols=self.data_cols["num_col"], trg_col="label")
        self.test_batch_loader = BatchGenerator(test_iterator, txt_col="text", num_cols=self.data_cols["num_col"], trg_col="label")

        logger = get_logger()
        logger.info("{} training examples are loaded.".format(len(train_data)))
        logger.info("{} validation examples are loaded.".format(len(valid_data)))
        logger.info("{} test examples are loaded.".format(len(test_data)))

