# Define a model
import torch
import torch.nn as nn
import numpy as np
from network.core.AlexNet import AlexNet

import click


class Model(nn.Module):

    def __init__(self):
        super(Model,self).__init__()

        self.conv0 = nn.Conv2d(1, 16, kernel_size=3, padding=5)
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3)

    def forward(self, x):
        h = self.conv0(x)
        h = self.conv1(h)
        return h

num_classes = 1000
model_alexnet = AlexNet(num_classes=num_classes)
model = Model()

modules = list(model.modules())

batch_size = 1
image_w = 227
image_h = 227

from estimator.sequential_model_estimator import ModuleStats

input_ = torch.randn(batch_size, 3, image_w, image_h)

ms = ModuleStats(model=model_alexnet, input=input_)

params = ms.get_params()
print("Params")
print(params)

output_sizes = ms.get_activations()
print("Output Sizes")
print(output_sizes)

param_bits = ms.get_param_bits()
print("Param Bits")
print(param_bits)

megabytes = 8.0 * 1024 * 1024

param_memory_mb = ms.get_param_bits() / megabytes
input_memory_mb = ms.get_input_bits() / megabytes
forward_memory_mb = ms.get_forward_bits() / megabytes
backward_memory_mb = ms.get_backward_bits() / megabytes

print("Param Memory MB : {}".format(param_memory_mb))
print("Input Memory MB : {}".format(input_memory_mb))
print("Forward Memory MB : {}".format(forward_memory_mb))
print("Backward Memory MB : {}".format(backward_memory_mb))

print("Total Memory MB : {}".format(ms.get_total_memory_mb()))

bits_vs_module = ms.get_param_bits_per_module()

print(bits_vs_module)
print(sum(bits_vs_module)/8/(1024**2))

width, _ = click.get_terminal_size()
click.echo('-' * width)
ms.get_memory_profile()
click.echo('-' * width)

# from fabulous.color import bold, magenta, highlight_green
#
# print(bold(magenta('hello world')))
#
# print(highlight_green('DANGER WILL ROBINSON!'))
#
# print(bold('hello') + ' ' + magenta(' world'))
#
# assert len(bold('test')) == 4


ms.get_memory_profile()