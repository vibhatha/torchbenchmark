import numpy as np
import torch as th
from torch import Tensor

# Reference: https://github.com/aayushmnit/Deep_learning_explorations/blob/master/1_MLP_from_scratch/Building_neural_network_from_scratch.ipynb



class Layer:
    """
    A building block. Each layer is capable of performing two things:

    - Process input to get output:           output = layer.forward(input)

    - Propagate gradients through itself:    grad_input = layer.backward(input, grad_output)

    Some layers also have learnable parameters which they update during layer.backward.
    """

    def __init__(self):
        """Here we can initialize layer parameters (if any) and auxiliary stuff."""
        # A dummy layer does nothing
        pass

    def forward(self, input: Tensor) -> Tensor:
        """
        Takes input data of shape [batch, input_units], returns output data [batch, output_units]
        """
        # A dummy layer just returns whatever it gets as input.
        return input

    def backward(self, input: Tensor, grad_output: Tensor):
        """
        Performs a backpropagation step through the layer, with respect to the given input.

        To compute loss gradients w.r.t input, we need to apply chain rule (backprop):

        d loss / d x  = (d loss / d layer) * (d layer / d x)

        Luckily, we already receive d loss / d layer as input, so you only need to multiply it by d layer / d x.

        If our layer has parameters (e.g. dense layer), we also need to update them here using d loss / d layer
        """
        # The gradient of a dummy layer is precisely grad_output, but we'll write it more explicitly

        num_units = input.shape[1]

        d_layer_d_input = th.eye(num_units)

        return th.dot(grad_output, d_layer_d_input)  # chain rule