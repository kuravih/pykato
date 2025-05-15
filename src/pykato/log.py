import logging


def setup_logger(name, terminator="\r"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.StreamHandler()
    handler.terminator = terminator
    handler.setFormatter(logging.Formatter("[%(asctime)15s] : %(levelname)-8s : %(message)s", datefmt="%Y%m%d.%H%M%S"))
    logger.addHandler(handler)
    return logger
