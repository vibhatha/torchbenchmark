# Define a model
import torch
import torch.nn as nn
import numpy as np
from network.core.AlexNet import AlexNet


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

# modules_alexnet = list(model_alexnet.modules())
# sequential_modules = []
# for module_id in range(1,len(modules_alexnet)):
#     sub_module = modules_alexnet[module_id]
#     if isinstance(sub_module, torch.nn.modules.container.Sequential):
#         sequential_modules.append(sub_module)

batch_size = 10
image_w = 227
image_h = 227

from estimator.sequential_model_estimator import ModuleExtractor
from estimator.sequential_model_estimator import ModuleStats

#me = ModuleExtractor(model=model_alexnet)
#seq_modules = me.get_sequential_modules()

# print(me)
#out_sizes = []

input_ = torch.randn(batch_size, 3, image_w, image_h)
#
# for id, mod in enumerate(seq_modules):
#     m = mod
#     out = m(input_)
#     out_sizes.append(np.array(out.size()))
#     input_ = out
#     print(id + 1, out.shape, m)

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


#Estimate Size
from estimator.pytorch_modelsize import SizeEstimator

se = SizeEstimator(model, input_size=(1, 1, 4, 4))

i = se.get_parameter_sizes()

se.estimate_size()

print(se.input_bits/megabytes, se.param_bits/megabytes, se.forward_backward_bits/megabytes)