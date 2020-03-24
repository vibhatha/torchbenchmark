##########################################################################
####### Reference: https://github.com/pytorch/pytorch/issues/2001#########
##########################################################################

import torch as th
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict

BYTES_PER_PARAM = 4
BYTE_TO_MEGA_BYTE_RATIO = 1024 ** 2


def network_summary(input_size, model, batch_size=None):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)

            m_key = '%s-%i' % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key]['input_shape'][0] = batch_size

            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, 'weight') and hasattr(module.weight, "size"):
                params += th.prod(th.LongTensor(list(module.weight.size())))
                if module.weight.requires_grad:
                    summary[m_key]['trainable'] = True
                else:
                    summary[m_key]['trainable'] = False
            if hasattr(module, 'bias') and hasattr(module.bias, "size"):
                params += th.prod(th.LongTensor(list(module.bias.size())))
            summary[m_key]['nb_params'] = params

        if not isinstance(module, nn.Sequential) and \
                not isinstance(module, nn.ModuleList) and \
                not (module == model):
            hooks.append(module.register_forward_hook(hook))

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(th.rand(1, *in_size)) for in_size in input_size]
    else:
        x = Variable(th.rand(1, *input_size))

    # create properties
    summary = OrderedDict()
    hooks = []
    # register hook
    model.apply(register_hook)
    # make a forward pass
    model(x)
    # remove these hooks
    for h in hooks:
        h.remove()

    return summary


import torch
from torchvision import models
from torchbench.network.core import AlexNet

network_name = "resnet152"

num_classes = 1000
model_alexnet = AlexNet(num_classes=num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg = models.resnet152().to(device)

input_size = (3, 227, 227)

sm1 = network_summary(input_size, vgg)

total = 0
trainable_params = 0
layers = []
input_shapes = []
output_shapes = []
param_items = []
param_items_bytes = []
trainable_param_items_bytes = []
trainable_params_list = []
row_data = []

delimiter = ":"

save_str = "Layer" + delimiter + "Input Shape" + delimiter + "Output Shape" + delimiter + "Num of Params" + delimiter + "Params (MB)" + delimiter + "Num of Trainable Params" + delimiter + "Trainable Params (MB)\n"

for layer in sm1:
    layer_name = layer
    input_shape = str(sm1[layer]["input_shape"])
    output_shape = str(sm1[layer]["output_shape"])
    params = sm1[layer]["nb_params"]
    layers.append(layer_name)
    input_shapes.append(input_shape)
    output_shapes.append(output_shape)
    trainable_param = 0

    if isinstance(params, torch.Tensor):
        params = params.item()

    param_items.append(params)
    params_in_bytes = params * BYTES_PER_PARAM / BYTE_TO_MEGA_BYTE_RATIO
    param_items_bytes.append(params_in_bytes)

    if "trainable" in sm1[layer]:
        if sm1[layer]["trainable"] == True:
            trainable_param = params
            trainable_params += trainable_param

    trainable_param_items_bytes.append(trainable_param)
    trainable_param_in_bytes = trainable_param * BYTES_PER_PARAM / BYTE_TO_MEGA_BYTE_RATIO
    total += params
    row_data.append(
        [layer_name, input_shape, output_shape, params, params_in_bytes, trainable_param, trainable_param_in_bytes])
    save_str += layer_name + delimiter + str(input_shape) + delimiter + str(output_shape) + delimiter + str(
        params) + delimiter + str(
        params_in_bytes) + delimiter + str(trainable_param) + delimiter + str(trainable_param_in_bytes) + "\n"

    # print(layer, str(input_shape), str(output_shape), str(params))

import termtables as tt

header = ["Layer", "Input Shape", "Output Shape", "Num of Params", "Params (MB)", "Num of Trainable Params",
          "Trainable Params (MB)"]
# data = [
#     [1, 2, 3], [613.23236243236, 613.23236243236, 613.23236243236]
# ]
print(row_data)

tt.print(
    row_data,
    header=header,
    style=tt.styles.ascii_thin,
    padding=(1, 1, 1, 1),
    alignment="ccccccc"
)

print("Total Param Memory : {} MB, Total Trainable Param Memory {} MB".format(total * 4 / (1024 ** 2),
                                                                              trainable_params * 4 / (1024 ** 2)))

with open("stats/" + network_name + ".info", "w") as fp:
    fp.write(save_str)
