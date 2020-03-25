import torch
from torchvision import models
from torchbenchmark.network.core import AlexNet


network_name = "VGG16"

num_classes = 1000
model_alexnet = AlexNet(num_classes=num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg = models.vgg16().to(device)

