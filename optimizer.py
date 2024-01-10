from typing import Dict
import numpy as np
from tensor import Tensor
from nn import NeuralNet

class Optimizer:
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError

#
#   Classic stochastic gradient descent
#

class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, net: NeuralNet) -> None:
        for param, grad, velocity in net.params_and_grads():
            param -= self.lr * grad 
            # param = param - lr*grad

#
#   SGD with Momentum
#

class Momentum(Optimizer): 
    def __init__(self, lr=0.1, beta=0.9, weight_decay=0.0005) -> None:
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay

    def step(self, net: NeuralNet) -> None:
        for param, grad, velocity in net.params_and_grads():
            velocity = self.beta * velocity - self.lr * grad - self.weight_decay * self.lr * param
            # v_{t+1} = b * v_t - LR * grad(param_t) - decay * LR * param_t
            param += velocity
            # param_{t+1} = param_t + velocity

#
#   RMSProp
#

class RMSProp(Optimizer):
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        self.lr = learning_rate
        self.beta = beta
        self.epsilon = epsilon


    def step(self, net: NeuralNet) -> None:
        for param, grad, velocity in net.params_and_grads():
            velocity = self.beta * velocity + (1 - self.beta) * (grad ** 2)
            # v_{t+1} = b * v_t - (1-b) * grad(param_t)^2
            param -= self.lr * (grad / (self.epsilon + np.sqrt(velocity)))
            # param_t+1 = param_t - lr * grad(param_t) / (epsilon + sqrt(velocity_{t+1}))
                       