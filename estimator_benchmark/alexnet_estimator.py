# Define a model
import torch
import torch.nn as nn
import numpy as np
from network.core.AlexNet import AlexNet


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.conv0 = nn.Conv2d(1, 64, kernel_size=3, padding=5)
        self.conv1 = nn.Conv2d(64, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3)

    def forward(self, x):
        h = self.conv0(x)
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)

        return h


num_classes = 1000
model_alexnet = AlexNet(num_classes=num_classes)
model = Model()

# Estimate Size
# from estimator.pytorch_modelsize import SizeEstimator
#
# se = SizeEstimator(model1, input_size=(3,3,10,10))
#
# i = se.get_parameter_sizes()
#
# print(i)


modules = list(model.modules())

# modules_alexnet = list(model_alexnet.modules())
# sequential_modules = []
# for module_id in range(1,len(modules_alexnet)):
#     sub_module = modules_alexnet[module_id]
#     if isinstance(sub_module, torch.nn.modules.container.Sequential):
#         sequential_modules.append(sub_module)

batch_size = 1
image_w = 128
image_h = 128

from estimator.sequential_model_estimator import ModuleExtractor

me = ModuleExtractor(model=model_alexnet)
seq_modules = me.get_sequential_modules()

# print(me)
out_sizes = []
input_ = torch.randn(batch_size, 3, image_w, image_h)
for id, mod in enumerate(seq_modules):
    m = mod
    out = m(input_)
    out_sizes.append(np.array(out.size()))
    input_ = out
    print(id + 1, out.shape, m)
