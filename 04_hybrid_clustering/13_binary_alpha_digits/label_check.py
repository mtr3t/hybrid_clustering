import numpy as np

def label_check(data, printer=False):
    '''
    this bit of code was proivided by dr.phillips and translated from r
    to order the lables to match the ground truth
    since the clustering methods used are unsupervised
    after the data is clustered there is no set assignment's for the labels
    we would like the clustering output to match the ground truth
    this function will flip the bits in order to match the ground truth
    ex: [0011] will return [0011]
        [1100] will return [0011]
    input:
        data: load the labels to be ordered (if needed)
    returns:
        data: labels ordered
    '''
    
    if printer: print('\nlabels to be checked:')
    if printer: print(data, '\n')
        
    f = np.median
    s = np.arange(np.min(data), np.max(data)+1)
    x = np.zeros(len(s))
    l = []
    
    for i in range(len(s)):
        l.append(np.where(data == s[i])[0])
        x[i] = f(l[i])
        
    ix = np.argsort(x)
    
    for i in range(len(s)):
        data[l[ix[i]]] = i
        
    if printer: print('labels returned:')
    if printer: print(data, '\n')
        
    return data