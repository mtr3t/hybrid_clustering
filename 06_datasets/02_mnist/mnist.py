import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt

from keras.datasets import mnist

# (train_X, train_y), (test_X, test_y) = mnist.load_data()

mnist = spio.loadmat('mnist_all.mat')

f = open('01_mnist.ds', 'w')
for i in range(10):
    images = mnist['train' + str(i)]/255
    for j in range(len(images)):
#     for j in range(1):
        temp = images[j]
        f.write(str(temp).strip('[]').replace('\n','').lstrip('  '))
        f.write('\n')
f.close()

f = open('01_mnist_ground_truth.ds', 'w')
for i in range(10):
    images = mnist['train' + str(i)]
    for j in range(len(images)):
        f.write(str(i))
        f.write('\n')
f.close()

f = open('02_mnist_0_and_1.ds', 'w')
for i in range(2):
    images = mnist['train' + str(i)]/255
    for j in range(len(images)):
#     for j in range(1):
        temp = images[j]
        f.write(str(temp).strip('[]').replace('\n','').lstrip('  '))
        f.write('\n')
f.close()

f = open('02_mnist_0_and_1_ground_truth.ds', 'w')
for i in range(2):
    images = mnist['train' + str(i)]
    for j in range(len(images)):
        f.write(str(i))
        f.write('\n')
f.close()

f = open('03_mnist_0_and_8.ds', 'w')
for i in 0,8:
    images = mnist['train' + str(i)]/255
    for j in range(len(images)):
#     for j in range(1):
        temp = images[j]
        f.write(str(temp).strip('[]').replace('\n','').lstrip('  '))
        f.write('\n')
f.close()

f = open('03_mnist_0_and_8_ground_truth.ds', 'w')
for i in 0,8:
    images = mnist['train' + str(i)]
    for j in range(len(images)):
        f.write(str(i))
        f.write('\n')
f.close()