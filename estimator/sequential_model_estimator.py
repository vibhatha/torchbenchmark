import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class ModuleExtractor(object):

    def __init__(self, model: nn.Module):
        self.model: nn.Module = model

    def get_sequential_modules(self):
        modules_alexnet = list(self.model.modules())
        sequential_modules = []
        for module_id in range(1, len(modules_alexnet)):
            sub_module = modules_alexnet[module_id]
            if isinstance(sub_module, torch.nn.modules.container.Sequential):
                sequential_modules.append(sub_module)
        self.sequential_modules = sequential_modules
        return sequential_modules

    def __str__(self):
        str1 = ""
        if isinstance(self.sequential_modules, list):
            for id, module in enumerate(self.sequential_modules):
                str1 += str(id + 1) + ":" + module.__str__() + "\n"
        return str1
