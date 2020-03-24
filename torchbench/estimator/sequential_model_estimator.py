import torch
import torch.nn as nn
import termtables as tt
import numpy as np


class ModuleExtractor(object):

    def __init__(self, model: nn.Module):
        self._model: nn.Module = model

    def get_sequential_modules(self) -> list:
        modules_alexnet = list(self._model.modules())
        sequential_modules = []
        for module_id in range(1, len(modules_alexnet)):
            sub_module = modules_alexnet[module_id]
            if isinstance(sub_module, torch.nn.modules.container.Sequential):
                sequential_modules.append(sub_module)
        self._sequential_modules = sequential_modules
        return sequential_modules

    def __str__(self) -> str:
        str1 = ""
        if isinstance(self._sequential_modules, list):
            for id, module in enumerate(self._sequential_modules):
                str1 += str(id + 1) + ":" + module.__str__() + "\n"
        return str1


class ModuleStats(object):

    def __init__(self, model: nn.Module, input: torch.Tensor = None, bit_size=32):
        self._model: nn.Module = model
        self._module_extractor = ModuleExtractor(model=self._model)
        self._modules: list = self._module_extractor.get_sequential_modules()
        self._input = input
        self._param_sizes = None
        self._param_size_per_module = []
        self._activation_size_per_module = []
        self._bit_size = bit_size
        self._input_bits = None
        self._param_bits = None
        self._param_bits_per_module = []
        self._forward_bits = None
        self._forward_bits_per_module = []
        self._backward_bits = None
        self._backward_bits_per_module = []
        self._bits_per_byte = 8
        self._total_bytes = 0
        self._total_megabytes = 0
        self._total_gigabytes = 0

    def get_modules(self):
        return self._modules

    def get_params(self) -> list:
        sizes = []
        sizes_per_module = []
        for id, module in enumerate(self._modules):
            sizes_per_sub_layer = []
            params = list(module.parameters())
            for param_id, param in enumerate(params):
                param_size = np.array(param.size())
                sizes_per_sub_layer.append(param_size)
                sizes.append(param_size)
            sizes_per_module.append(sizes_per_sub_layer)
        self._param_sizes = sizes
        self._param_size_per_module = sizes_per_module
        return sizes

    def get_activations(self) -> list:
        input_: torch.Tensor = self._input
        activation_sizes: list = []
        for id, mod in enumerate(self._modules):
            out = mod(input_)
            activation_sizes.append(np.array(out.size()))
            input_ = out
        self._activation_size_per_module = activation_sizes
        return activation_sizes

    def _calculate_param_bits_per_module(self):
        bits_per_module_list = []
        if self._param_size_per_module is None:
            self.get_params()

        for module_id, param_module in enumerate(self._param_size_per_module):
            param_values = param_module
            if len(param_values) >= 1:
                bits_per_module = 0
                for param_sub_value_id, param_sub_value in enumerate(param_values):
                    # print(module_id, param_sub_value_id, param_sub_value)
                    bits_per_sub_module = np.prod(np.array(param_sub_value)) * self._bit_size
                    bits_per_module += bits_per_sub_module
                bits_per_module_list.append(bits_per_module)
            else:
                # adding zero bits
                bits_per_module_list.append(0)

        self._param_bits_per_module = bits_per_module_list
        return self._param_bits_per_module

    def _calculate_param_bits(self):
        total_bits = 0
        if self._param_sizes is None:
            self._param_sizes = self.get_params()

        for param_id, param in enumerate(self._param_sizes):
            bits = np.prod(np.array(param)) * self._bit_size
            #print(param_id, param, bits)
            total_bits += bits
        self._param_bits = total_bits
        return total_bits

    def get_param_bits(self):
        return self._calculate_param_bits()

    def get_param_bits_per_module(self):
        return self._calculate_param_bits_per_module()

    def get_input_bits(self):
        _input_size = self._input.size()
        self._input_bits = np.prod(np.array(_input_size)) * self._bit_size
        return self._input_bits

    def _calculate_forward_bits(self):
        total_bits = 0
        for activation_id, activation in enumerate(self._activation_size_per_module):
            bits = np.prod(np.array(activation)) * self._bit_size
            total_bits += bits
        # Forward computation activations are saved to be used in the backward
        self._forward_bits = total_bits
        return self._forward_bits

    def _calculate_backward_bits(self):
        # Backward computation uses the saved activations and generate new memory
        # generated new memory is theoretically equal to the all parameter memory
        self._backward_bits = self.get_param_bits()
        return self._backward_bits

    def _format_table(self, modules_info: list = [], memory_info: list = []):
        header = ["Layer Num","Layer Description", "Memory"]
        row_all_data = []
        id = 1
        for module, memory in zip(memory_info, memory_info):
            row_all_data.append([id, module, memory])
            id = id + 1
        #print(row_all_data)
        # tt.print(
        #     row_all_data,
        #     header=header,
        #     style=tt.styles.ascii_thin,
        #     padding=(1, 1, 1),
        #     alignment="ccc"
        # )


    def _fuse_layer_info_with_memory(self):
        self._format_table(modules_info=self._modules, memory_info=self._param_bits_per_module)
        # for module, module_memory in zip(self._modules, self._param_bits_per_module):
        #     str_module: str = module.__str__()
        #     str_module_memory: str = str(module_memory)
        #     str_item_module = str_module.split("\n")

    def get_forward_bits(self):
        return self._calculate_forward_bits()

    def get_backward_bits(self):
        return self._calculate_backward_bits()

    def get_memory_profile(self):
        self._fuse_layer_info_with_memory()

    def _memory_profile(self):
        self.get_params()
        self.get_activations()
        self.get_param_bits()
        self.get_input_bits()
        self.get_forward_bits()
        self.get_backward_bits()

        total = self._param_bits + self._forward_bits + self._backward_bits + self._input_bits

        self._total_bytes = (total / self._bits_per_byte)
        self._total_megabytes = self._total_bytes / (1024 ** 2)
        self._total_gigabytes = self._total_bytes / (1024 ** 3)

    def get_total_memory_bytes(self):
        self._memory_profile()
        return self._total_bytes

    def get_total_memory_mb(self):
        self._memory_profile()
        return self._total_megabytes

    def get_total_memory_gb(self):
        self._memory_profile()
        return self._total_gigabytes
