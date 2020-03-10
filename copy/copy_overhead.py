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

#kb_sizes = np.array([1]) * KB
#mb_sizes = np.array([1024]) * MB


def experiments(size_type=kb_sizes, data_type=KB, data_type_name="KB", copy_device1='cuda:1', copy_device2='cuda:2',
                reps=20):
    generation_time_per_size_mean = []
    tensor_conversion_time_per_size_mean = []
    cpu_gpu_copy_time_per_size_mean = []
    gpu_to_gpu_copy_time_per_size_mean = []
    gpu_cpu_copy_time_per_size_mean = []

    generation_time_per_size_std = []
    tensor_conversion_time_per_size_std = []
    cpu_gpu_copy_time_per_size_std = []
    gpu_to_gpu_copy_time_per_size_std = []
    gpu_cpu_copy_time_per_size_std = []

    for size in size_type:
        generation_time = []
        tensor_conversion_time = []
        cpu_gpu_copy_time = []
        gpu_to_gpu_copy_time = []
        gpu_cpu_copy_time = []
        for rep in range(reps):
            elements = int(size / byte_size)
            # generate numpy array and time
            t1 = time.time()
            array = np.arange(elements, dtype='d')
            t2 = time.time()
            generation_time.append(t2 - t1)

            # convert numpy to torch tensor and time
            t3 = time.time()
            tensor = torch.from_numpy(array)
            t4 = time.time()
            tensor_conversion_time.append(t4 - t3)

            # copy from cpu to gpu device 1
            t5 = time.time()
            torch.cuda.synchronize(copy_device1)
            tensor1 = tensor.to(copy_device1)
            torch.cuda.synchronize(copy_device1)
            #print("Data @Device1 : ",tensor1.device)
            t6 = time.time()
            cpu_gpu_copy_time.append(t6 - t5)

            # copy from gpu device 1 to 2
            t7 = time.time()
            torch.cuda.synchronize(copy_device1)
            tensor2 = tensor1.to(copy_device2)
            torch.cuda.synchronize(copy_device2)
            tensor2 = tensor2 + tensor2
            #print("Data @Device2 : ", tensor2.device)
            t8 = time.time()
            gpu_to_gpu_copy_time.append(t8 - t7)

            # copy from gpu to cpu
            t9 = time.time()
            torch.cuda.synchronize(copy_device1)
            tensor3 = tensor1.to('cpu')
            torch.cuda.synchronize(copy_device1)
            t10 = time.time()
            gpu_cpu_copy_time.append(t10 - t9)
            print(
                "Rep {} : Data Size {} {}, Type {}, Gen {}, Conv {}, CPU->GPU0 {}, GPU0->GPU1 {}, GPU1->CPU {}".format(
                    rep, int(array.nbytes / data_type), data_type_name,
                    array.dtype, (t2 - t1), (t4 - t3), (t6 - t5), (t8 - t7), (t10 - t9)))

        # calculate avg time for repititions
        generation_times = np.array(generation_time[int(reps / 2):reps])
        tensor_conversion_times = np.array(tensor_conversion_time[int(reps / 2):reps])
        cpu_gpu_copy_times = np.array(cpu_gpu_copy_time[int(reps / 2):reps])
        gpu_to_gpu_copy_times = np.array(gpu_to_gpu_copy_time[int(reps / 2):reps])
        gpu_cpu_copy_times = np.array(gpu_cpu_copy_time[int(reps / 2):reps])

        generation_time_avg = generation_times.mean()
        tensor_conversion_time_avg = tensor_conversion_times.mean()
        cpu_gpu_copy_time_avg = cpu_gpu_copy_times.mean()
        gpu_to_gpu_copy_time_avg = gpu_to_gpu_copy_times.mean()
        gpu_cpu_copy_time_avg = gpu_cpu_copy_times.mean()

        generation_time_std = generation_times.std()
        tensor_conversion_time_std = tensor_conversion_times.std()
        cpu_gpu_copy_time_std = cpu_gpu_copy_times.std()
        gpu_to_gpu_copy_time_std = gpu_to_gpu_copy_times.std()
        gpu_cpu_copy_time_std = gpu_cpu_copy_times.std()

        # populate data size by time list
        generation_time_per_size_mean.append(generation_time_avg)
        tensor_conversion_time_per_size_mean.append(tensor_conversion_time_avg)
        cpu_gpu_copy_time_per_size_mean.append(cpu_gpu_copy_time_avg)
        gpu_to_gpu_copy_time_per_size_mean.append(gpu_to_gpu_copy_time_avg)
        gpu_cpu_copy_time_per_size_mean.append(gpu_cpu_copy_time_avg)

        generation_time_per_size_std.append(generation_time_std)
        tensor_conversion_time_per_size_std.append(tensor_conversion_time_std)
        cpu_gpu_copy_time_per_size_std.append(cpu_gpu_copy_time_std)
        gpu_to_gpu_copy_time_per_size_std.append(gpu_to_gpu_copy_time_std)
        gpu_cpu_copy_time_per_size_std.append(gpu_cpu_copy_time_std)

        print(generation_time_avg, tensor_conversion_time_avg, cpu_gpu_copy_time_avg, gpu_to_gpu_copy_time_avg,
              gpu_cpu_copy_time_avg)

    return (generation_time_per_size_mean, generation_time_per_size_std), (
        tensor_conversion_time_per_size_mean, tensor_conversion_time_per_size_std), (
               cpu_gpu_copy_time_per_size_mean, cpu_gpu_copy_time_per_size_std), (
               gpu_to_gpu_copy_time_per_size_mean, gpu_to_gpu_copy_time_per_size_std), (
               gpu_cpu_copy_time_per_size_mean, gpu_cpu_copy_time_per_size_std)


print("KB Experiments")
(generation_time_kb_mean, generation_time_kb_std), (tensor_conversion_time_kb_mean,
                                                    tensor_conversion_time_kb_std), (
    cpu_gpu_copy_time_kb_mean, cpu_gpu_copy_time_kb_std), (gpu_to_gpu_copy_time_kb_mean, gpu_to_gpu_copy_time_kb_std), (
    gpu_cpu_copy_time_kb_mean, gpu_cpu_copy_time_kb_std) = experiments()

print("MB Experiments")
(generation_time_mb_mean, generation_time_mb_std), (tensor_conversion_time_mb_mean,
                                                    tensor_conversion_time_mb_std), (
    cpu_gpu_copy_time_mb_mean, cpu_gpu_copy_time_mb_std), (gpu_to_gpu_copy_time_mb_mean, gpu_to_gpu_copy_time_mb_std), (
    gpu_cpu_copy_time_mb_mean, gpu_cpu_copy_time_mb_std) = experiments(
    size_type=mb_sizes, data_type=MB, data_type_name="MB")

total_data_mean = np.concatenate((generation_time_kb_mean, tensor_conversion_time_kb_mean, cpu_gpu_copy_time_kb_mean,
                                  gpu_to_gpu_copy_time_kb_mean, gpu_cpu_copy_time_kb_mean), axis=0)

total_data_std = np.concatenate((generation_time_kb_std, tensor_conversion_time_kb_std, cpu_gpu_copy_time_kb_std,
                                 gpu_to_gpu_copy_time_kb_std, gpu_cpu_copy_time_kb_std), axis=0)

np.savetxt('total_times_mean_kb.csv', total_data_mean, delimiter=',')
np.savetxt('total_times_std_kb.csv', total_data_std, delimiter=',')

print("---------KB----------")
print(total_data_mean)
print("---------------------")

# MB Data

total_data_mean = np.concatenate((generation_time_mb_mean, tensor_conversion_time_mb_mean, cpu_gpu_copy_time_mb_mean,
                                  gpu_to_gpu_copy_time_mb_mean, gpu_cpu_copy_time_mb_mean), axis=0)

total_data_std = np.concatenate((generation_time_mb_std, tensor_conversion_time_mb_std, cpu_gpu_copy_time_mb_std,
                                 gpu_to_gpu_copy_time_mb_std, gpu_cpu_copy_time_mb_std), axis=0)

print("---------MB----------")
print(total_data_mean)
print("---------------------")
# print(total_data_std)

np.savetxt('total_times_mean_mb.csv', total_data_mean, delimiter=',')
np.savetxt('total_times_std_mb.csv', total_data_std, delimiter=',')
