import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from network.core.AlexNet import AlexNet

num_classes = 1000
num_batches = 10
batch_size = 120
image_w = 128
image_h = 128
num_repeat = 20

cuda_available = torch.cuda.is_available()

print("===================================================")
print("Cuda Available : {}".format(cuda_available))
print("===================================================")


def train(model, epochs=10):
    model.train(True)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    one_hot_indices = torch.LongTensor(batch_size) \
        .random_(0, num_classes) \
        .view(batch_size, 1)

    for epoch in range(epochs):
        for _ in range(num_batches):
            # generate random inputs and labels
            inputs = torch.randn(batch_size, 3, image_w, image_h)
            if cuda_available:
                inputs = inputs.to('cuda:0')
            labels = torch.zeros(batch_size, num_classes) \
                .scatter_(1, one_hot_indices, 1)

            # run forward pass
            optimizer.zero_grad()

            outputs = model(inputs)

            # run backward pass
            labels = labels.to(outputs.device)
            loss_fn(outputs, labels).backward()
            optimizer.step()


import time



stats = []


def cuda_test():
    model_cpu_init_time = []
    model_cpu_gpu_copy_time = []
    model_train_time = []

    for i in range(10):
        t1 = time.time()
        model = AlexNet(num_classes=num_classes)
        t2 = time.time()
        model_cpu_init_time.append(t2 - t1)

        torch.cuda.synchronize('cuda:0')
        t3 = time.time()
        model = model.to('cuda:0')
        torch.cuda.synchronize('cuda:0')
        t4 = time.time()
        model_cpu_gpu_copy_time.append(t4 - t3)

        torch.cuda.synchronize('cuda:0')
        t5 = time.time()
        train(model)
        t6 = time.time()
        torch.cuda.synchronize('cuda:0')
        model_train_time.append(t6 - t5)

    model_cpu_init_time = np.array(model_cpu_init_time)
    model_cpu_gpu_copy_time = np.array(model_cpu_gpu_copy_time)
    model_train_time = np.array(model_train_time)

    model_cpu_init_time_mean, model_cpu_init_time_std = np.mean(model_cpu_init_time), np.std(model_cpu_init_time)

    model_cpu_gpu_copy_time_mean, model_cpu_gpu_copy_time_std = np.mean(model_cpu_gpu_copy_time), np.std(
        model_cpu_gpu_copy_time)

    train_time_mean, train_time_std = np.mean(model_train_time), np.std(model_train_time)

    print("Single Node Model Init Time (CPU):", model_cpu_init_time_mean)
    print("Single Node Model Copy CPU->GPU:0:", model_cpu_gpu_copy_time_mean)
    print("Single Node Training Time:", train_time_mean)

inputs = torch.randn(batch_size, 3, image_w, image_h)
print(inputs.size())

model = AlexNet(num_classes=num_classes)

model(inputs)

cuda_test()