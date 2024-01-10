"""
Neural nets are made of layers. Each needs to pass its
inputs forward, and gradient backpropagate. 

Ex:

inputs -> Linear -> Tanh -> Linear -> output
"""
from typing import Dict, Callable, Tuple, Any
import numpy as np
from tensor import Tensor
from scipy import signal # type: ignore

class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}
        self.velocity: Dict[str, Any] = {}

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
        self.velocity["w"] = 0
        self.velocity["b"] = 0
    
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
    

class Convolution(Layer):
    def __init__(self, input_shape: Tuple[int, int, int], kernel_size: int, depth: int): # input_shape 3D tensor
        super().__init__()
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.params["k"] = np.random.randn(*self.kernels_shape)
        self.params["b"] = np.random.randn(*self.output_shape)
        self.velocity["k"] = 0
        self.velocity["b"] = 0

    def forward(self, input: Tensor):
        self.input = input
        self.output = np.copy(self.params["b"])
        kernel = self.params["k"]
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], kernel[i, j], "valid")
        return self.output

    def backward(self, output_gradient):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        kernel = self.params["k"]
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], kernel[i,j], "full")
        
        self.grads["k"] = kernels_gradient
        self.grads["b"] = output_gradient
        return input_gradient
    

class Dense(Layer):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.params["w"] = np.random.randn(output_size, input_size)
        self.params["b"] = np.random.randn(output_size, 1)
        self.velocity["w"] = 0
        self.velocity["b"] = 0

    def forward(self, input: Tensor):
        self.input = input
        return np.dot(self.params["w"], self.input) + self.params["b"]

    def backward(self, output_gradient: Tensor):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.params["w"].T, output_gradient)
        self.grads["w"] = weights_gradient
        self.grads["b"] = output_gradient
        return input_gradient


class Reshape(Layer):
    def __init__(self, input_shape: Tuple, output_shape: Tuple):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input: Tensor):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient: Tensor):
        return np.reshape(output_gradient, self.input_shape)


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
    return np.where(x <= 0, c, 1).astype(Tensor)

class Leaky_ReLU(Activation):
    def __init__(self) -> None:
        super().__init__(leaky_relu, leaky_relu_prime)


# elu activation layer:

def elu(x: Tensor, alpha = 1):  # alpha is 1
    return np.where(x > 0, x, alpha(np.exp(x) - 1))  # *alpha

def elu_prime(x: Tensor):
	return np.where(x > 0, 1, elu(x) + 1)

class eLU(Activation):
    def __init__(self) -> None:
        super().__init__(elu, elu_prime)

# softmax activation layer:

def sm(x: Tensor, a=None, derivative=False):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def sm_prime(x: Tensor):
	return 1

class softmax(Activation):
    def __init__(self) -> None:
        super().__init__(sm, sm_prime)