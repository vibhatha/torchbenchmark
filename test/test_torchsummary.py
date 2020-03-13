import torch
from torchvision import models
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg = models.alexnet().to(device)

summary(vgg, (3, 227, 227))