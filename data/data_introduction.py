import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

CIFAR_DIR = "./cifar-10-batches-py"
print(os.listdir(CIFAR_DIR))

with open(os.path.join(CIFAR_DIR, "data_batch_1"), 'rb') as f:
    data = pickle.load(f, encoding='bytes')
    print(type(data))
    print(data.keys())
    print(type(data[b'data']))
    print(type(data[b'labels']))
    print(type(data[b'batch_label']))
    print(type(data[b'filenames']))
    print(data[b'data'].shape)
    print(data[b'data'][0:2])
    print(data[b'labels'][0:2])
    print(data[b'batch_label'])
    print(data[b'filenames'][0:2])

image_arr = data[b'data'][100]
image_arr = image_arr.reshape((3, 32, 32)) # 32 32 3
image_arr = image_arr.transpose((1, 2, 0))
plt.imshow(image_arr)