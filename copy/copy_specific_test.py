import torch
import numpy as np
import time

byte_size = np.arange(1).nbytes
KB = 1024
MB = KB * 1024
GB = MB * 1024
message_sizes = []
kb_sizes = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]) * KB
mb_sizes = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]) * MB

gb1 = mb_sizes[10]

print(gb1)

data = np.arange(gb1)

arr_size_gb = data.nbytes / GB / byte_size
print(arr_size_gb)
time1 = []
time2 = []
time3 = []
time4 = []
for i in range(10):
    tensor = torch.tensor(data)
    t1 = time.time()
    tensor1 = tensor.to('cuda:0')
    time1.append(time.time() - t1)

    t1 = time.time()
    tensor1.to('cuda:1')
    time2.append(time.time() - t1)

    t1 = time.time()
    tensor1.to('cuda:3')
    time3.append(time.time() - t1)

    t1 = time.time()
    tensor1.to('cpu')
    time4.append(time.time() - t1)

cpu_gpu = sum(time1)/len(time1)

gpu_gpu1 = sum(time2)/len(time1)

gpu_gpu2 = sum(time3)/len(time1)

gpu_cpu = sum(time4)/len(time1)

print("CPU to GPU : {}".format(cpu_gpu))

print("GPU 0 to GPU 1: {}".format(gpu_gpu1))

print("GPU 0 to GPU 7: {}".format(gpu_gpu2))

print("GPU to CPU : {}".format(gpu_cpu))

