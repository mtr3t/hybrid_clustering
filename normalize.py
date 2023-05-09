import numpy as np

def normalize(matrix, printer=False):
    '''
    normalize data between 0 and 1 for for plotting alphas
    '''
    
    alphas = (matrix-np.min(matrix))/(np.max(matrix)-np.min(matrix))
    
    if printer: print('alphas =')
    if printer: print(alphas, '\n')
        
    return alphas