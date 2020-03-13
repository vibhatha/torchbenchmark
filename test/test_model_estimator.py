import torch
from torchvision import models
from network.core.AlexNet import AlexNet

network_name = "resnet152"

num_classes = 1000
model_alexnet = AlexNet(num_classes=num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg = models.alexnet().to(device)

input_size = (3, 227, 227)

import estimator.model_estimator as est

mest = est.ModelEstimator(name='alexnet', model=vgg, input_size=input_size, save=True, save_path='stats/alexnet_info_dummy.info')

mest.generate_summary()