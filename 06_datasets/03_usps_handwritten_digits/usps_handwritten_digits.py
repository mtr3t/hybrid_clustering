import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File('usps.h5', 'r') as hf:
    train = hf.get('train')
    X_tr = train.get('data')[:]
    y_tr = train.get('target')[:]
    test = hf.get('test')
    X_te = test.get('data')[:]
    y_te = test.get('target')[:]
    
# all usps (7291)
    f = open('01_usps.ds', 'w')
for i in range(len(X_tr)):
    f.write(str(X_tr[i]).strip('[]').replace('\n',''))
    f.write('\n')
f.close()

f = open('01_usps_ground_truth.ds', 'w')
for i in range(len(y_tr)):
    f.write(str(y_tr[i]))
    f.write('\n')
f.close()

# same as mnist (1404)
f = open('02_usps_1404.ds', 'w')
for i in range(1404):
    f.write(str(X_tr[i]).strip('[]').replace('\n',''))
    f.write('\n')
f.close()

f = open('02_usps_1404_ground_truth.ds', 'w')
for i in range(1404):
    f.write(str(y_tr[i]))
    f.write('\n')
f.close()

# same as mnist (468)
f = open('03_usps_468.ds', 'w')
for i in range(468):
    f.write(str(X_tr[i]).strip('[]').replace('\n',''))
    f.write('\n')
f.close()

f = open('03_usps_468_ground_truth.ds', 'w')
for i in range(468):
    f.write(str(y_tr[i]))
    f.write('\n')
f.close()

# same as mnist (180)
f = open('04_usps_180.ds', 'w')
for i in range(180):
    f.write(str(X_tr[i]).strip('[]').replace('\n',''))
    f.write('\n')
f.close()

f = open('04_usps_180_ground_truth.ds', 'w')
for i in range(180):
    f.write(str(y_tr[i]))
    f.write('\n')
f.close()