import torch
from torchvision import models

network_name = "resnet152"

num_classes = 1000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg = models.alexnet().to(device)



import torchbenchmark.estimator.model_estimator as est

input_size = (3, 227, 227)
mest = est.ModelEstimator(name='alexnet', model=vgg, input_size=input_size, batch_size=2 ,save=True, save_path='stats/alexnet_info_dummy.info')

mest.generate_summary()