import unittest
from pykato.buffer import NumpyCircularBuffer
import numpy as np

class TestNumpyCircularBuffer(unittest.TestCase):
    # pylint: disable=missing-class-docstring

    def test_NumpyCircularBuffer(self):
        buffer = NumpyCircularBuffer(4, dtype=np.dtype([("probe", "i4"), ("data", np.complex64, (10,10))]))
        self.assertIsInstance(buffer, NumpyCircularBuffer, "Buffer could not be created")

    def test_NumpyCircularBuffer_push_back(self):
        buffer = NumpyCircularBuffer(10, dtype=float)
        self.assertIsInstance(buffer, NumpyCircularBuffer, "Buffer could not be created")

        test_data = np.random.rand(10)
        for index, value in enumerate(test_data):
            buffer.push_back(value)
            self.assertEqual(buffer.queue.tolist(), test_data[:index+1].tolist(), "Queues not equal")

    def test_NumpyCircularBuffer_pop_front(self):
        buffer = NumpyCircularBuffer(10, dtype=float)
        self.assertIsInstance(buffer, NumpyCircularBuffer, "Buffer could not be created")

        test_data = np.random.rand(6)
        for index, value in enumerate(test_data):
            buffer.push_back(value)

        elements = buffer.pop_front()
        self.assertEqual(elements.tolist(), test_data[:1].tolist(), "Queues not equal")
        self.assertEqual(buffer.queue.tolist(), test_data[1:].tolist(), "Queues not equal")

        elements = buffer.pop_front(2)
        self.assertEqual(buffer.queue.tolist(), test_data[3:].tolist(), "Queues not equal")

