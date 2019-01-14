"""
Nerual network made up of layers
Ech layers need to have its inputs foraward
and propogate error backwards



"""
from typing import Dict, Callable

import numpy as np
from myNet.tensor import Tensor

class Layer:

    def __init__(self)-> None:
        self.param: Dict[str, Tensor] = {}
        self.grad: Dict[str, Tensor] = {}


    def forward(self, inputs: Tensor)-> Tensor:
        """
        Produce the outputs of the inputs

        """
        raise NotImplementedError
    def backward(self, grad: Tensor)-> Tensor:
        """
        Backpropogates the gradient through the layer
        """
        raise NotImplementedError

class Linear(Layer):
    """
    COmputes output= input @ W + b
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        # inputs will be [batch_size, input_size]
        # output wiill be [batch_size, outtu_size]
        super.__init__()
        self.param["w"] = np.random.randn(input_size,output_size)
        self.param["b"] = np.random.randn(input_size)
    def forward(self, inputs: Tensor) -> Tensor:
        """
        output = input @ w + b
        """
        self.inputs = inputs
        return inputs @ self.param["w"] + self.params["b"]
    
    def backward(self, grad: Tensor) -> Tensor
        """
        if y = f(x) and x = a * b + c
        then dy/da = f'(x) * b
        then dy/db = f'(x) * a
        then dy/dc = f'(x) 

        if y = f(x) and x = a @ b + c
        then dy/da = f'(x) * b.T
        then dy/db = a.T * f'(x) 
        then dy/dc = f'(x) 
        """
        self.grad["b"] = np.sum(grad, axis=0)
        self.grad["w"] = self.input.T @ grad
        return grad @ self.params["w"].T

F = callable[[Tensor], Tensor]

class Activation(layer):
    """
    Element wise operation to inputs
    """
    def __init__(self, f: F, f_prime: F) -> none:
        super().__init__()
        self.f = f
        self.f_prime = f_prime
    def forward(self, inputs: Tensor) -> Tensor:
        self.input = inputs
        return self.f(input)
    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)
        """
        return self.f_prime(self.inputs) * grad

def tanh(x: Tensor)-> Tensor:
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)
    return 1-y**2

class Tanh(Activation):

    def __init__(self):
        super()__init__(tanh, tanh_prime)