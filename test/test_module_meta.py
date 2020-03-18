import torch
from torchvision import models
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg = models.alexnet().to(device)

for name, layer in vgg.named_children():
    print("###########################################")
    print("Name: ", name)
    print("Layer: ", layer)
    print("============================================")