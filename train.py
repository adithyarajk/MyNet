"""
Function that can train our neural network

"""

from tensor import Tensor
from nn import NeuralNet
from loss import Loss, MSE
from optimizers import Optimizer, SGD
from data import DataIterator, BatchIterator


def train(net: NeuralNet,
    inputs: Tensor,
    targets: Tensor,
    num_epochs: int =5000,
    iterator: DataIterator= BatchIterator(),
    loss = MSE(),
    Optimizer: Optimizer = SGD()
        ) -> None:

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            predicted = net.forward(batch.inputs)
            epochs += loss.loss(predicted, batch_targets)
            net.bacward(grad)
            optimizer.step(net)
        print(epoch, epoch_loss)
