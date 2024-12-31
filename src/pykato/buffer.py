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
        buffer = NumpyQueue(5, dtype = np.dtype([("rep", "i4"), ("data", np.complex64, (200,200))]))

    """

    def __init__(self, max_size: int = 1, dtype=np.float32):
        self.dtype = dtype
        self.max_size = max_size
        self._queue = np.zeros(self.max_size, dtype=self.dtype)
        self.front = 0
        self.rear = -1
        self.size = 0

    def is_empty(self) -> bool:
        """
        Check if the queue empty.
        """
        return self.size == 0

    def is_full(self) -> bool:
        """
        Check if the queue is full.
        """
        return self.size == self.max_size

    def empty(self):
        """
        Empty the queue
        """
        self.front = 0
        self.rear = -1
        self.size = 0
        self._queue = np.empty(self.max_size, dtype=self.dtype)

    def push_back(self, element):
        """
        Insert an element in the back.
        """
        if self.is_full():
            raise OverflowError("Queue is full")
        self.rear = (self.rear + 1) % self.max_size
        self._queue[self.rear] = element
        self.size += 1

    def pop_front(self):
        """
        Retrieve an element from the front.
        """
        if self.is_empty():
            raise IndexError("Queue is empty")
        element = self._queue[self.front]
        self.front = (self.front + 1) % self.max_size
        self.size -= 1
        return element

    @property
    def latest(self):
        """
        latest element
        """
        return self._queue[self.rear]

    @property
    def queue(self) -> np.ndarray:
        """
        element queue
        """
        if self.is_empty():
            return np.empty(self.max_size, dtype=self.dtype)
        if self.rear >= self.front:
            return self._queue[self.front : self.rear + 1]
        else:
            return np.concatenate((self._queue[self.front :], self._queue[: self.rear + 1]), axis=0)
