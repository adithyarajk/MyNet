"""

A Neural network is just collecion of layers
behaves a lot like a layer

"""

from typing import Sequence, Iterator, Tuple

from tensor import Tensor
from layers import Layer

class NeuralNet:
    def __init__(self, layers: Sequence[Layer] ) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            input = layer.forward(inputs)
        return inputs

    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params_and_grad(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad
