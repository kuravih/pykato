import numpy as np


class NumpyCircularBuffer:
    """
    A Numpy CircularBuffer

    Parameters:
        max_size: int = 1
            Size of the queue
        dtype: np.float32
            Data type of the queue

    Attributes:
        queue: np.ndarray
            The underlying numpy array.

    Methods:
        is_empty(): bool
            Check if the queue empty.
        is_full(): bool
            Check if the queue is full.
        empty():
            Empty the queue
        push_back(element):
            Insert an element in the back.
        pop_front():
            Retrieve an element from the front.

    Usage:
        buffer = NumpyCircularBuffer(5, dtype = np.dtype([("rep", "i4"), ("data", np.complex64, (200,200))]))

    """

    def __init__(self, max_size: int = 1, dtype=np.float32):
        self.dtype = dtype
        self._max_size = max_size
        self._queue = np.empty(self._max_size, dtype=self.dtype)
        self._front = 0
        self._back = -1
        self._size = 0

    @property
    def max_size(self) -> int:
        return self._max_size

    def is_empty(self) -> bool:
        """
        Check if the queue empty.
        """
        return self._size == 0

    def is_full(self) -> bool:
        """
        Check if the queue is full.
        """
        return self._size == self._max_size

    def empty(self):
        """
        Empty the queue
        """
        self._front = 0
        self._back = -1
        self._size = 0

    def push_back(self, element):
        """
        Insert an element in the back.
        """
        if self.is_full():
            raise OverflowError("Queue is full")
        self._back = (self._back + 1) % self._max_size
        self._queue[self._back] = element
        self._size = self._size + 1

    def pop_front(self, count: int = 1):
        """
        Retrieve an element from the front.
        """
        if self.is_empty():
            raise IndexError("Queue is empty")
        elements = self._queue[self._front : self._front + count]
        self._front = (self._front + count) % self._max_size
        self._size -= count
        return elements

    @property
    def back(self) -> int:
        return self._back

    @property
    def latest(self):
        """
        Latest element in the buffer.
        """
        if self.is_empty():
            raise IndexError("Buffer is empty")
        return self._queue[self._back]

    @property
    def queue(self) -> np.ndarray:
        """
        element queue
        """
        if self.is_empty():
            return np.array([], dtype=self.dtype)  # Return an empty array instead of a preallocated empty one
        if self._back >= self._front:
            return self._queue[self._front : self._back + 1]
        return np.concatenate((self._queue[self._front :], self._queue[: self._back + 1]), axis=0)
