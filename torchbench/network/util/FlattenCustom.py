import torch
import torch.nn as nn


class FlattenCustom(nn.Module):

    def __init__(self):
        super(FlattenCustom, self).__init__()

    def forward(self, x):
        x = torch.flatten(x, 1)
        return x
