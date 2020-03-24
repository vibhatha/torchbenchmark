import torch
from torchvision import models
from torchbench.network.core import AlexNet
import click

network_name = "VGG16"

num_classes = 1000
model_alexnet = AlexNet(num_classes=num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vgg = models.vgg16().to(device)

import torchbench.estimator.model_estimator as est

input_sizes = [(3, 64, 64)]  # , (3, 128, 128), (3, 227, 227), (3, 256, 256), (3, 512, 512), (3, 1024, 1024)]
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
exp_config = []
total_input_memory_list = []
total_activation_memory_list = []
total_parameters_memory_list = []

for input_size_id, input_size in enumerate(input_sizes):
    for batch_id, batch_size in enumerate(batch_sizes):
        print(batch_id, input_size_id, input_size, batch_size)
        exp_cur_config = (input_size, batch_size)
        exp_config.append(exp_cur_config)
        mest = est.ModelEstimator(name='alexnet', model=vgg, input_size=input_size, batch_size=batch_size, save=False,
                                  save_path='stats/alexnet_info_dummy.info', console=False)
        mest.generate_summary()
        total_input_memory_list.append(mest.total_input_memory)
        total_activation_memory_list.append(mest.total_activation_parameters_memory)
        total_parameters_memory_list.append(mest.total_parameters_memory)

width, _ = click.get_terminal_size()
click.echo('-' * width)
for id, (config, input_memory, activation_memory, parameter_memory) in enumerate(
        zip(exp_config, total_input_memory_list, total_activation_memory_list, total_parameters_memory_list)):
    print(id, config, input_memory, activation_memory, parameter_memory)

data = [total_input_memory_list, total_activation_memory_list, total_parameters_memory_list]
legend = ['Total Input Memory', 'Total Activation Memory', 'Total Parameter Memory']

from zyplots.core.LineChart import LineChart

lc = LineChart(data=data, legend=legend, title= network_name + ' Memory Profile', xaxis_title='Minibatch Size',
               yaxis_title='Memory (MB)', xticks=batch_sizes,
               fig_save_path='figures/' + network_name + '_memory_profile.png')
lc.plot()
