
import numpy as np
import random
import torch
import yaml

from tqdm import trange
from utils.data import Dataset
from utils.model import instantiate_model
from utils.logger import get_logger

if __name__ == '__main__':

    args = yaml.load(open('./configs/runner.yml', 'r'), Loader=yaml.FullLoader)
    args = {**args, **yaml.load(open('./configs/data.yml', 'r'), Loader=yaml.FullLoader)}

    seed = args["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_file, test_file, embedding_source = args["train_file"], args["test_file"], args["embedding_source"]

    dataset = Dataset(data_columns=args["data_info"], batch_size=args["batch_size"], seq_len=args["seq_len"])
    dataset.load_data(train_file, test_file, embedding_source)

    vocab_size = len(dataset.vocab)
    embeddings = dataset.embeddings

    model = instantiate_model(args["model"], vocab_size, embeddings)

    # tqdm_range = trange(args["epochs"], desc="Epoch", unit="epoch")
    print("=" * 54)
    for epoch in range(args["epochs"]):
        _, _ = model.run_epoch(dataset.train_batch_loader, dataset.valid_batch_loader)
        print("-" * 54)
        train_acc = model.evaluate(dataset.train_batch_loader)
        val_acc = model.evaluate(dataset.valid_batch_loader)
        print("Epoch %3d ended | train acc.: %3.2f | val acc.: %.2f" % ((epoch + 1), train_acc * 100, val_acc * 100))
        # tqdm_range.set_postfix_str("Loss: %g" % loss)
        print("=" * 54)

    train_acc = model.evaluate(dataset.train_batch_loader)
    valid_acc = model.evaluate(dataset.valid_batch_loader)
    test_acc = model.evaluate(dataset.test_batch_loader)

    logger = get_logger()
    logger.info('Training Accuracy: %' + '%3.2f.' % (train_acc * 100))
    logger.info('Validation Accuracy: %' + '%3.2f.' % (valid_acc * 100))
    logger.info('Test Accuracy: %' + '%3.2f.' % (test_acc * 100))

