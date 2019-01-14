"""
Well feed inputs to our netwokr into batches
tools fo iterating over data in batches

"""

from typing import Iterator, NamedTuple
import numpy as numpy

from myNet.tensor import Tensor

Batch = NamedTuple("Batch", [("inputs", Tensor)], ("targets"), Tensor)

class DataIterator:
    def __cal__(self, inputs: Tensor, targets: Tensor) -> Iterator:
        raise NotImplementedError

class BatchIterator:
    def __init__(self, batch_size: int =32, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __cal__(self, inputs: Tensor, targets: Tensor) -> Iterator:
        starts = np.arrange(0, len(inputs), self.batch_size)
        if self.shuffle:
            np.random.shuffle(starts)

        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start: end]

            yield Batch(batch_inputs, batch_targets)