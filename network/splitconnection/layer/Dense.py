import torch as th
from torch import Tensor
import numpy as np
from network.splitconnection.core.Layer import Layer

# Reference: https://github.com/aayushmnit/Deep_learning_explorations/blob/master/1_MLP_from_scratch/Building_neural_network_from_scratch.ipynb


class Dense(Layer):
    def __init__(self, input_units: int, output_units: int, learning_rate: float = 0.1):
        """
        A dense layer is a layer which performs a learned affine transformation:
        f(x) = <W*x> + b
        """
        self.learning_rate = learning_rate
        self.weights = th.Tensor(np.random.normal(loc=0.0,
                                        scale=np.sqrt(2 / (input_units + output_units)),
                                        size=(input_units, output_units)))
        self.biases = th.Tensor(np.zeros(output_units))
        #print(self.weights.shape, self.biases.shape)

    def forward(self, input: Tensor) -> Tensor:
        """
        Perform an affine transformation:
        f(x) = <W*x> + b

        input shape: [batch, input_units]
        output shape: [batch, output units]
        """
        #print("Dense:Forward ",input.shape, self.weights.shape, self.biases.shape)
        #(32, 784) (784, 100) (100,)
        # https://github.com/spro/practical-pytorch/issues/54
        # torch.dot(hidden.view(-1), energy.view(-1))
        Wx = th.mm(input, self.weights)
        return Wx + self.biases

    def backward(self, input: Tensor, grad_output: Tensor) -> Tensor:
        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        grad_input: Tensor = th.mm(grad_output, self.weights.T)

        # compute gradient w.r.t. weights and biases
        grad_weights: Tensor = th.mm(input.T, grad_output)
        grad_biases = grad_output.mean(axis=0) * input.shape[0]

        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape

        # Here we perform a stochastic gradient descent step.
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input
