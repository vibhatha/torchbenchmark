import torch
from torch import Tensor
from typing import Dict, List, Tuple

logs = []
logs2 = []
device_info: Dict[int, str] = {}
device_count = 8

micro_batch_count = 16
micro_batches: List[Tensor] = []

grad_data: Dict[int, Tuple[str, Tensor]]


class GlobalContext:
    number: int = 0
    number2: int = 0
    log: List[int] = {}
    log2: List[int] = {}


_global_context = GlobalContext()

def populate_microbatches(micro_batches: List[Tensor], count: int) -> None:
    for id in range(count):
        micro_batches.append(torch.FloatTensor([id]))


def show_microbatches(micro_batches: List[Tensor]):
    for t in micro_batches:
        print(t)


def populate_device(device_info: Dict[int, str], device_count: int) -> None:
    for id in range(device_count):
        device_info[id] = [id, 'cuda:' + str(id)]


def show_devices(devices: Dict[int, str]) -> None:
    for id in range(len(devices)):
        print(*devices[id])


class Log(torch.autograd.Function):

    @staticmethod
    def forward(ctx, number, tensor):
        _global_context.number = number
        return tensor.detach()

    @staticmethod
    def backward(ctx, grad):
        _global_context.log = logs
        logs.append(_global_context.number)
        return None, grad


class Log2(torch.autograd.Function):
    @staticmethod
    def forward(ctx: GlobalContext, number, tensor):
        _global_context.number2 = number
        return tensor.detach()

    @staticmethod
    def backward(ctx: GlobalContext, grad):
        logs2.append(_global_context.number2)
        _global_context.log2 = logs2
        return None, grad


# initialize
a = torch.rand(1, device='cpu', requires_grad=True)
b = torch.rand(1, device='cpu', requires_grad=True)

populate_device(device_info=device_info, device_count=device_count)
# show_devices(devices=device_info)

populate_microbatches(micro_batches=micro_batches, count=micro_batch_count)
# show_microbatches(micro_batches=micro_batches)

# Do forward
b = Log2.apply(2, b)
a = Log.apply(1, a)


# Do backward
b.backward()
a.backward()



print(logs)
print("-------------------")
print(logs2)

print("=======================")
print(_global_context.number)
print(_global_context.log)
print(_global_context.number2)
print(_global_context.log2)
print("=======================")
