import numpy as np

def symmetrize(coeff, printer):
    # Symmetrize
    coeff = coeff + coeff.T
    if printer: print('symmetrize coeff =')
    if printer: print(coeff, '\n')
    return coeff