import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt

from keras.datasets import mnist

# (train_X, train_y), (test_X, test_y) = mnist.load_data()

mnist = spio.loadmat('mnist_all.mat')

# all 60000 images
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

# all 1's and 0's
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

# all 0's and 8's
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

# 39 each to equal binary alpha digits
f = open('04_mnist_1400.ds', 'w')
for i in range(10):
    images = mnist['train' + str(i)]/255
    for j in range(140):
#     for j in range(1):
        temp = images[j]
        f.write(str(temp).strip('[]').replace('\n','').lstrip('  '))
        f.write('\n')
f.close()

f = open('04_mnist_1400_ground_truth.ds', 'w')
for i in range(10):
    images = mnist['train' + str(i)]
    for j in range(140):
        f.write(str(i))
        f.write('\n')
f.close()

# 78 0's and 1's
f = open('05_mnist_0_and_1_78.ds', 'w')
for i in range(2):
    images = mnist['train' + str(i)]/255
    for j in range(39):
#     for j in range(1):
        temp = images[j]
        f.write(str(temp).strip('[]').replace('\n','').lstrip('  '))
        f.write('\n')
f.close()

f = open('05_mnist_0_and_1_78_ground_truth.ds', 'w')
for i in range(2):
    images = mnist['train' + str(i)]
    for j in range(39):
        f.write(str(i))
        f.write('\n')
f.close()

# 78 0's and 8's
f = open('06_mnist_0_and_8_78.ds', 'w')
for i in 0,8:
    images = mnist['train' + str(i)]/255
    for j in range(39):
#     for j in range(1):
        temp = images[j]
        f.write(str(temp).strip('[]').replace('\n','').lstrip('  '))
        f.write('\n')
f.close()

f = open('06_mnist_0_and_8_78_ground_truth.ds', 'w')
for i in 0,8:
    images = mnist['train' + str(i)]
    for j in range(39):
        f.write(str(i))
        f.write('\n')
f.close()

# 47 each to equal binary alpha digits
f = open('07_mnist_470.ds', 'w')
for i in range(10):
    images = mnist['train' + str(i)]/255
    for j in range(47):
#     for j in range(1):
        temp = images[j]
        f.write(str(temp).strip('[]').replace('\n','').lstrip('  '))
        f.write('\n')
f.close()

f = open('07_mnist_470_ground_truth.ds', 'w')
for i in range(10):
    images = mnist['train' + str(i)]
    for j in range(47):
        f.write(str(i))
        f.write('\n')
f.close()

# 18 each to equal binary alpha digits
f = open('08_mnist_180.ds', 'w')
for i in range(10):
    images = mnist['train' + str(i)]/255
    for j in range(18):
#     for j in range(1):
        temp = images[j]
        f.write(str(temp).strip('[]').replace('\n','').lstrip('  '))
        f.write('\n')
f.close()

f = open('08_mnist_180_ground_truth.ds', 'w')
for i in range(10):
    images = mnist['train' + str(i)]
    for j in range(18):
        f.write(str(i))
        f.write('\n')
f.close()