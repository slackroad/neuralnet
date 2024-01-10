# Neural Net

A Deep Learning library made from scratch with just Python and Numpy. Gradients are fully automated, both forward and backward propagation.
Deriving the math from scratch by hand using only online resources and textbooks helped me understand the math deeper, and how to take advantage of deep learning.

## Installation
```sh
python3 -m pip install git+git@github.com:slackroad/neuralnet.git
```


## Usage

An example below using the MNIST dataset:


### Import modules

```python
from keras.datasets import mnist # type: ignore
from keras import utils # type: ignore
import numpy as np
from optimizer import RMSProp
from optimizer import SGD
from layers import Dense
from loss import BinaryCrossEntropy
from train import train
from layers import Convolution, Reshape, Sigmoid

from nn import NeuralNet
from tensor import Tensor
```

### Make Sequential Model

After preprocessing data, we can create a Neural Net from an array of layers:

```python
# neural network
network = NeuralNet([
    Convolution((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 10),
    Sigmoid()
])

```

### Then, we can train the model using CrossEntropy and RMSProp

```python
train(
    net = network,
    inputs = x_train,
    targets = y_train,
    loss = CrossEntropy(),
    optimizer = RMSProp(),
    num_epochs=5000
)
```

### Optimizers:

* SGD             
* SGD with Momentum 
* Rmsprop         

### Layers:

* Conv2D
* Linear
* Reshape
* Dense             
* Activation

### Loss functions:

* CrossEntropy
* MSE

### Activation Functions:

* sigmoid
* relu
* leakyRelu
* elu
* tanh
