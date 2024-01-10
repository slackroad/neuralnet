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

def preprocess_data(x: Tensor, y: Tensor, limit):
    x, y = x[:limit], y[:limit]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = utils.to_categorical(y)
    y = y.reshape(len(y), 10, 1)
    return x, y

# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 300)
x_test, y_test = preprocess_data(x_test, y_test, 100)

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

# train
train(
    net = network,
    inputs = x_train,
    targets = y_train,
    loss = BinaryCrossEntropy(),
    optimizer = RMSProp(),
    num_epochs=5000
)

# test
total = 0
for x, y in zip(x_test, y_test):
    output = network.forward(x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
    correct = (np.argmax(output) == np.argmax(y))
    if (correct): 
        total += 1
print("Accuracy: ", float(total)/ float(len(x_test)))
