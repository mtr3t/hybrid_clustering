import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt

bad = spio.loadmat('binaryalphadigs.mat', squeeze_me=True)

images = bad['dat']
nclasses = bad['numclass']
classlabels = bad['classlabels']
classcounts = bad['classcounts']

# all data from the binary alpha digits dataset (1404 images)
f = open('01_binary_alpha_digits_1404.ds', 'w')
for i in range(len(images)):
    for j in range(len(images[i])):
        temp = images[i][j].flatten()
        f.write(str(temp).strip('[]').replace('\n',''))
        f.write('\n')
f.close()

f = open('01_binary_alpha_digits_1404_ground_truth.ds', 'w')
for i in range(len(images)):
    for j in range(len(images[i])):
        f.write(str(i))
        f.write('\n')
f.close()

# just the 0's and 1's (78 images)
f = open('02_binary_alpha_digits_0_and_1_78.ds', 'w')
for i in range(2):
    for j in range(len(images[i])):
        temp = images[i][j].flatten()
        f.write(str(temp).strip('[]').replace('\n',''))
        f.write('\n')
f.close()

f = open('02_binary_alpha_digits_0_and_1_78_ground_truth.ds', 'w')
for i in range(2):
    for j in range(len(images[i])):
        f.write(str(i))
        f.write('\n')
f.close()

# just the 0's and 8's (78 images)
f = open('03_binary_alpha_digits_0_and_8_78.ds', 'w')
for i in 0, 8:
    for j in range(len(images[i])):
        temp = images[i][j].flatten()
        f.write(str(temp).strip('[]').replace('\n',''))
        f.write('\n')
f.close()

f = open('03_binary_alpha_digits_0_and_8_78_ground_truth.ds', 'w')
for i in 0,8:
    for j in range(len(images[i])):
        f.write(str(i))
        f.write('\n')
f.close()

# smaller subset of data only 13 digits each
f = open('04_binary_alpha_digits_468.ds', 'w')
for i in range(len(images)):
    for j in range(13):
        temp = images[i][j].flatten()
        f.write(str(temp).strip('[]').replace('\n',''))
        f.write('\n')
f.close()

f = open('04_binary_alpha_digits_468_ground_truth.ds', 'w')
for i in range(len(images)):
    for j in range(13):
        f.write(str(i))
        f.write('\n')
f.close()

# smaller subset of data only 5 digits each
f = open('05_binary_alpha_digits_180.ds', 'w')
for i in range(len(images)):
    for j in range(5):
        temp = images[i][j].flatten()
        f.write(str(temp).strip('[]').replace('\n',''))
        f.write('\n')
f.close()

f = open('05_binary_alpha_digits_180_ground_truth.ds', 'w')
for i in range(len(images)):
    for j in range(5):
        f.write(str(i))
        f.write('\n')
f.close()