import numpy as np

def normalize(matrix, printer):
    ## Normalize for plotting
    alphas = (matrix-np.min(matrix))/(np.max(matrix)-np.min(matrix))
    if printer: print('alphas =')
    if printer: print(alphas, '\n')
    return alphas