import numpy as np

def adjacency(coeff, set_K, A, printer):
    if set_K == 0:
        K = np.shape(A)[1]
    else:
        if set_K > np.shape(A)[1]:
            print('Max K =', np.shape(A)[1], ', please set K to a lower value.')
        else:
            K = set_K
    if printer: print('K =', K, '\n')

    # Select the top K coefficients
    newcoeff = np.zeros(np.shape(coeff))
    if printer: print('newcoeff =')
    if printer: print(newcoeff, '\n')
    
    if printer: print('coeff =')
    if printer: print(coeff, '\n')
        
    indices = np.apply_along_axis(lambda x: np.argsort(x)[::-1],1,coeff)[:,range(K)]
    if printer: print('indices =')
    if printer: print(indices, '\n')

    for x in range(np.shape(coeff)[0]):
        newcoeff[x,indices[x,:]] = coeff[x,indices[x,:]]
    if printer: print('newcoeff =')
    if printer: print(newcoeff, '\n')
    
    return newcoeff