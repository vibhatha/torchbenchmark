import torch as th
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
import termtables as tt

_BYTES_PER_PARAM = 4
_BYTE_TO_KILO_BYTE_RATIO = 1024
_BYTE_TO_MEGA_BYTE_RATIO = 1024 ** 2
_BYTE_TO_GIGA_BYTE_RATIO = 1024 ** 3


################################################################
######################### References:###########################
#                                                              #
# 1. https://github.com/pytorch/pytorch/issues/2001            #
# 2. https://github.com/jacobkimmel/pytorch_modelsize          #
# 3. https://github.com/sksq96/pytorch-summary                 #
################################################################

class ModelEstimator(object):

    def __init__(self, name="", model=None, input_size=(3, 227, 227), batch_size=None, bytes_per_param: int = 4,
                 converter: str = 'MB', delimiter="!",
                 header=["Layer", "Input Shape", "Output Shape", "Num of Params", "Params (MB)",
                         "Num of Trainable Params", "Trainable Params (MB)"], save: bool = False,
                 console: bool = True, summary: bool = True, save_path: str = None):
        self._name = name
        self._input_size = input_size
        self._model = model
        self._batch_size = batch_size
        self._bytes_per_param = bytes_per_param
        self._converter = converter
        self._delimiter = delimiter
        self._header = header
        self._save = save
        self._console = console
        self._summary = summary
        self._save_path = save_path

    def network_summary(self):
        model = self._model
        batch_size = self._batch_size
        input_size = self._input_size

        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split('.')[-1].split("'")[0]
                module_idx = len(summary)

                m_key = '%s-%i' % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]['input_shape'] = list(input[0].size())
                summary[m_key]['input_shape'][0] = batch_size
                # For complex networks this check is important
                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [
                        [-1] + list(o.size())[1:] for o in output
                    ]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = batch_size

                params = 0
                # SOME NETWORKS HAS COMPLEX ARCHITRCTURES
                # Testing has weight and size parameter check is important
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

    def generate_summary(self):

        _CONVERSION_VALUE = _BYTE_TO_KILO_BYTE_RATIO

        if self._converter == 'MB':
            _CONVERSION_VALUE = _BYTE_TO_MEGA_BYTE_RATIO
        if self._converter == 'GB':
            _CONVERSION_VALUE = _BYTE_TO_GIGA_BYTE_RATIO

        if self._input_size is None or self._model is None:
            raise Exception("Input Size {} or Model {} is not defined".format(self._input_size, self._model))

        sm1 = self.network_summary()

        total = 0
        trainable_params = 0
        layers = []
        input_shapes = []
        output_shapes = []
        param_items = []
        param_items_bytes = []
        trainable_param_items_bytes = []
        trainable_params_list = []
        # append for table print
        row_data = []

        save_str = ""

        '''
        Generate Header of the stats file
        Use default delimiter or any other symbol other than "[" or ","  or "("
        "," Is used within internal data structures, it could complicate file reading with Excel, LibCalc, etc
        '''

        if self._delimiter == ",":
            raise Exception(
                " ',' Is used within internal data structures, it could complicate file reading with Excel, LibCalc, etc")

        if self._save:
            for col_id, col_name in enumerate(self._header):
                save_str += col_name + self._delimiter
            save_str += "\n"

        id: int = 1
        print("Id : {}".format(id))
        id += 1
        for layer in sm1:
            layer_name = layer
            input_shape = str(sm1[layer]["input_shape"])
            output_shape = str(sm1[layer]["output_shape"])
            params = sm1[layer]["nb_params"]
            layers.append(layer_name)
            input_shapes.append(input_shape)
            output_shapes.append(output_shape)
            trainable_param = 0

            if isinstance(params, th.Tensor):
                params = params.item()

            param_items.append(params)
            params_in_bytes = params * self._bytes_per_param / _CONVERSION_VALUE
            param_items_bytes.append(params_in_bytes)

            if "trainable" in sm1[layer]:
                if sm1[layer]["trainable"] == True:
                    trainable_param = params
                    trainable_params += trainable_param

            trainable_param_items_bytes.append(trainable_param)
            trainable_param_in_bytes = trainable_param * self._bytes_per_param / _CONVERSION_VALUE
            total += params
            row_data.append(
                [layer_name, input_shape, output_shape, params, params_in_bytes, trainable_param,
                 trainable_param_in_bytes])

            if self._save:
                save_str += layer_name + self._delimiter + str(input_shape) + self._delimiter + str(
                    output_shape) + self._delimiter + str(
                    params) + self._delimiter + str(
                    params_in_bytes) + self._delimiter + str(trainable_param) + self._delimiter + str(
                    trainable_param_in_bytes) \
                            + "\n"

            alignment = ""

        if self._console:
            for _, _ in enumerate(self._header):
                alignment += "c"

            print(len(self._header))
            tt.print(
                row_data,
                header=self._header,
                style=tt.styles.ascii_thin,
                padding=(1, 1, 1, 1),
                alignment=alignment
            )

        if self._console:
            if self._summary:
                print("Total Param Memory : {} {}, Total Trainable Param Memory {} {}".format(
                    total * 4 / _CONVERSION_VALUE,
                    self._converter,
                    trainable_params * 4 / _CONVERSION_VALUE,
                    self._converter
                ))

        if self._save:
            if self._save_path:
                with open(self._save_path, "w") as fp:
                    fp.write(save_str)
            else:
                raise Exception("Save Path not specified")