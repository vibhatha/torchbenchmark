from torch import Tensor
from torchbenchmark.network.splitconnection.core import Layer
#Reference: https://github.com/aayushmnit/Deep_learning_explorations/blob/master/1_MLP_from_scratch/Building_neural_network_from_scratch.ipynb


class ReLU(Layer):
    def __init__(self):
        """ReLU layer simply applies elementwise rectified linear unit to all inputs"""
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Apply elementwise ReLU to [batch, input_units] matrix"""
        relu_forward = input.clamp(min=0)
        return relu_forward

    def backward(self, input: Tensor, grad_output: Tensor) -> Tensor:
        """Compute gradient of loss w.r.t. ReLU input"""
        relu_grad = input > 0
        return grad_output * relu_grad
