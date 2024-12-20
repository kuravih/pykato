import unittest
from pykato.buffer import NumpyCircularBuffer
import numpy as np

class TestNumpyCircularBuffer(unittest.TestCase):
    # pylint: disable=missing-class-docstring

    def test_NumpyCircularBuffer(self):
        buffer = NumpyCircularBuffer(4, dtype=np.dtype([("probe", "i4"), ("data", np.complex64, (10,10))]))
        self.assertIsInstance(buffer, NumpyCircularBuffer, "Buffer could not be created")
