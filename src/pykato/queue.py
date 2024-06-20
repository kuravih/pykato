import numbers
from typing import Optional, Union, Tuple
import numpy as np

class NumpyQueue:
    def __init__(self, max_size, element_shape:Optional[Union[int, Tuple[int, ...]]]=1):
        self.max_size = max_size
        self.element_shape = element_shape
        if isinstance(element_shape, int):
            self.element_shape = (element_shape,)
        self.queue = np.zeros((max_size,) + self.element_shape, dtype=np.float32)
        self.front = 0
        self.rear = -1
        self.size = 0

    def is_empty(self):
        return self.size == 0

    def is_full(self):
        return self.size == self.max_size

    def enqueue(self, element):
        if (not isinstance(element, numbers.Number)) and (element.shape != self.element_shape):
            raise ValueError(f"Element shape must be {self.element_shape} but its {element.shape}")
        if self.is_full():
            raise OverflowError("Queue is full")
        self.rear = (self.rear + 1) % self.max_size
        self.queue[self.rear] = element
        self.size += 1

    def dequeue(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        element = self.queue[self.front]
        self.front = (self.front + 1) % self.max_size
        self.size -= 1
        return element

    def get_queue(self):
        if self.is_empty():
            return np.empty((0,) + self.element_shape, dtype=np.float32)
        if self.rear >= self.front:
            return self.queue[self.front:self.rear + 1]
        else:
            return np.concatenate((self.queue[self.front:], self.queue[:self.rear + 1]), axis=0)
