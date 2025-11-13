import unittest
from pykato.log import setup_logger

logger = setup_logger("test_log", terminator="\n")


class TestLog(unittest.TestCase):
    # pylint: disable=missing-class-docstring

    def test_log_info(self):
        logger.info("info message")

    def test_log_warning(self):
        logger.warning("warning message")

    def test_log_critical(self):
        logger.critical("critical message")

    def test_log_error(self):
        logger.error("error message")
