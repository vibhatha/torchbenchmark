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

elements = int(gb1 / byte_size)

data = np.arange(elements, dtype='d')

arr_size_gb = data.nbytes / GB
print(arr_size_gb)
time1 = []
time2 = []
time3 = []
time4 = []
reps = 20
print("Rep,CPU->GPU,GPU0->GPU1,GPU0->GPU3,GPU0->CPU")
for i in range(reps):
    data = np.arange(elements, dtype='d')

    tensor = torch.from_numpy(data)

    t1 = time.time()
    tensor1 = tensor.to('cuda:0')
    t2 = time.time()
    time1.append(t2 - t1)

    t3 = time.time()
    tensor2 = tensor1.to('cuda:1')
    t4 = time.time()
    time2.append(t4 - t3)

    t5 = time.time()
    tensor3 = tensor1.to('cuda:3')
    t6 = time.time()
    time3.append(t6 - t5)

    t7 = time.time()
    tensor = tensor1.to('cpu')
    t8 = time.time()
    time4.append(t8 - t7)
    # print("Rep {}, CPU->GPU {}, GPU0->GPU1 {}, GPU0->GPU3 {}, GPU0->CPU {}".format(i, (t2 - t1), (t4 - t3), (t6 - t5),
    #                                                                                (t8 - t7)))
    #
    print("{},{},{},{},{}".format(i, (t2 - t1), (t4 - t3), (t6 - t5), (t8 - t7)))

cpu_gpu = sum(time1[int(reps/2):reps]) / len(time1) / 2

gpu_gpu1 = sum(time2[int(reps/2):reps]) / len(time1) / 2

gpu_gpu2 = sum(time3[int(reps/2):reps]) / len(time1) / 2

gpu_cpu = sum(time4[int(reps/2):reps]) / len(time1) / 2

print("CPU to GPU : {}".format(cpu_gpu))

print("GPU 0 to GPU 1: {}".format(gpu_gpu1))

print("GPU 0 to GPU 7: {}".format(gpu_gpu2))

print("GPU to CPU : {}".format(gpu_cpu))
