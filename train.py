"""
Function to train neural network
"""
from tensor import Tensor
from nn import NeuralNet
from loss import Loss, MSE
from optimizer import Optimizer, SGD
from data import DataIterator, BatchIterator

def train(net: NeuralNet, 
          inputs: Tensor,
          targets: Tensor,
          num_epochs: int = 5000,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = MSE(),
          optimizer: Optimizer = SGD()) -> None:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            for (input, target) in zip(batch.inputs, batch.targets):
                predicted = net.forward(input)
                epoch_loss += loss.loss(predicted, target)
                grad = loss.grad(predicted, target)
                net.backward(grad)
                optimizer.step(net)
        print(epoch, epoch_loss)

