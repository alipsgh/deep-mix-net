
import logging


def get_logger(name=None):

    # logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    return logging.getLogger(name)

