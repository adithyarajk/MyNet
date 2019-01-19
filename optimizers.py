"""
Used to adjust paramerters of our networks based
on gradient computed during backpropogation

"""
from nn import NeuralNet

class Optimizer:

    def step(self, net: NeuralNet ) -> None:
        raise NotImplementedError

class SGD(Optimizer):

    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr
    
    def step(self, net: NeuralNet) -> None:
        for param, grad in net.params_and_grad():
            param -= self.lr*grad
        
