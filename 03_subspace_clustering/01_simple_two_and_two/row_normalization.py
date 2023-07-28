import numpy as np

def row_normalizer(coeff, printer):
    # Normalize each row - not needed but doesn't hurt on most examples
    # rows = 1, col = 0
    coeff = coeff / np.apply_along_axis(np.max,1,coeff)[:,None]
    if printer: print('norm of coeff =')
    if printer: print(coeff, '\n')
    return coeff