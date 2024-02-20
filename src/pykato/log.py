import logging

stream_handler = logging.StreamHandler()
stream_handler.terminator = "\r"
logging.basicConfig(format="[%(asctime)15s] : %(levelname)-8s : %(message)s", level=logging.INFO, datefmt="%Y%m%d.%H%M%S", force=True, handlers=[stream_handler])
