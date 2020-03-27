import torch
from torchvision import models

from torchbenchmark.network.core.unet import UNet

network_name = "unet"

num_classes = 1000

net = UNet(n_channels=3, n_classes=num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg = models.alexnet().to(device)



import torchbenchmark.estimator.model_estimator as est

input_size = (3, 192, 192)
mest = est.ModelEstimator(name='unet', model=net, input_size=input_size, batch_size=8 ,save=True, save_path='stats/unet_mb_1_info_dummy.info')

mest.generate_summary()