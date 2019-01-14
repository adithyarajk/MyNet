"""
A Loss function measures how good ourp presdictions are


"""


from myNet.tensor import Tensor

class Loss:
    """

    """

    def loss(self, predicted: Tensor, actual: Tensor)-> float:
        return np.sum((predicted-actual)**2)

    def grad(self, predicted: Tensor, actual: Tensor)-> Tensor:
        return 2*(predicted-actual)