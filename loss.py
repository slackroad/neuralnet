from tensor import Tensor

import numpy as np

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError
    
class MSE(Loss):
    """
    MSE is mean squared, we're doing total squared error
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual)**2)
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2*(predicted - actual)
    
class BinaryCrossEntropy(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        print("predicted: ", predicted)
        print("actual: ", actual)
        return np.mean(-actual * np.log(predicted) - (1 - actual) * np.log(1 - predicted))
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return ((1 - actual) / (1 - predicted) - actual / predicted) / np.size(actual)
    