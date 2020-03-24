import torch
from typing import Dict, Tuple
from torch import Tensor
from torch.autograd import Variable

_global_itr = 0


class MyReLUF(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input: Tensor, epoch: int):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        output = input.clamp(min=0)
        output = output.to('cuda:1')
        new_input = input.to('cuda:1')
        ctx.save_for_backward(new_input)
        data_map_f[_global_itr] = ['My_relu.forward', (input, None)]

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        pass


class MyReLUB(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input: Tensor):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        return input

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        #input, = ctx.saved_tensors

        #epoch:Tensor = d[1]
        input = data_map_f[_global_itr][1][0]

        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

dtype = torch.float
device = torch.device("cpu")

# data_map[0] : Oth iteration with ['layer-name:0', (0))]
data_map: Dict[str, Tuple[Tensor, Tensor]] = {}

data_map_f: Dict[str, Tuple[Tensor, Tensor]] = {}
data_map_b: Dict[str, Tuple[Tensor, Tensor]] = {}
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs.
x: Tensor = torch.randn(N, D_in, device=device, dtype=dtype)
y: Tensor = torch.randn(N, D_out, device=device, dtype=dtype)

# Create random Tensors for weights.
w1: Tensor = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2: Tensor = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    _global_itr = t
    # To apply our Function, we use Function.apply method. We alias this as 'relu'.
    reluF: torch.autograd.Function = MyReLUF.apply
    reluB: torch.autograd.Function = MyReLUB.apply

    # Forward pass: compute predicted y using operations; we compute
    # ReLU using our custom autograd operation.
    t_var: Tensor = Variable(torch.Tensor([t]).type(torch.IntTensor), requires_grad=False)
    y_pred: Tensor = reluF(x.mm(w1)).mm(w2)
    y_pred_b: Tensor = reluB(x.mm(w1)).mm(w2)

    y_pred_b.data = y_pred.data

    # Compute and print loss
    loss = (y_pred_b - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # Use autograd to compute the backward pass.
    loss.backward()

    # Update weights using gradient descent
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()



print(data_map_f[0][0])
print(data_map_f[0][1][0].size())
print(data_map_f[0][1][1])

