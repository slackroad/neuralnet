"""
Neural nets are made of layers. Each needs to pass its
inputs forward, and gradient backpropagate. 

Ex:

inputs -> Linear -> Tanh -> Linear -> output
"""
from typing import Dict, Callable
import numpy as np
from tensor import Tensor

class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Produce outputs corresponding to inputs
        """
        raise NotImplementedError
    
    def backward(self, grad: Tensor) -> Tensor:
        """
        Backpropagate gradient through layer
        """
        raise NotImplementedError
    

class Linear(Layer):
    """
    computes out = inputs @ w + b
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        # inputs will be (batch_size, input_size)
        # outputs will be (batch_size, output_size)
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)
    
    def forward(self, inputs: Tensor) -> Tensor:
        """
        out = inputs @ w + b
        """
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]
    
    def backward(self, grad: Tensor) -> Tensor:
        """
        If y = f(x), x = a @ b + c
        Then,
        dy/da = f'(x) @ b^T
        dy/db = a^T @ f'(x)
        dy/dc = f'(x)
        """
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T
    
F = Callable[[Tensor], Tensor]

class Activation(Layer):
    """
    Applies a function elementwise to its inputs 
    """
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)
    
    def backward(self, grad: Tensor) -> Tensor:
        """
        If y= f(x), x = g(z), then 
        dy/dz = f'(x) * g'(z) (Chain Rule)
        """
        return self.f_prime(self.inputs) * grad

# Tanh activation layer:

def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y ** 2
    
class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)


# Sigmoid activation layer:

def sigmoid(x: Tensor) -> Tensor:
    # 1/(1+e^{-x})
    return 1/(1 + np.exp(-x))

def sigmoid_prime(x: Tensor) -> Tensor:
    # sigmoid(x) ( 1- sigmoid(x))
    y = sigmoid(x)
    return y * (1-y)

class Sigmoid(Activation):
    def __init__(self) -> None:
        super().__init__(sigmoid, sigmoid_prime)


# Relu activation layer:

def relu(x: Tensor) -> Tensor:
    return np.maximum(0, x)

def relu_prime(x: Tensor) -> Tensor:
    return np.where(x <= 0, 0, 1).astype(Tensor)


class ReLU(Activation):
    def __init__(self) -> None:
        super().__init__(relu, relu_prime)

# Leaky-Relu activation layer:

def leaky_relu(x: Tensor) -> Tensor: # default leaky constant is 0.01
    return np.where(x <= 0, 0.01 * x, x).astype(Tensor)

def leaky_relu_prime(x: Tensor, c: float= 0.01) -> Tensor:
    return np.where(x <= 0, 0.01, 1).astype(Tensor)

class Leaky_ReLU(Activation):
    def __init__(self) -> None:
        super().__init__(leaky_relu, leaky_relu_prime)


