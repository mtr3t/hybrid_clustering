import numpy as np

def indices(A, set_K, printer):
    
    if set_K > np.shape(A)[1]:
        print('Max K =', np.shape(A)[1], ', please set K to a lower value.')
    else:
        K = set_K
    if printer: print('K =', K, '\n')

    N = A.shape[0]
    if printer: print('N =', N, '\n')
    
    # Select the top K coefficients
    newcoeff = np.abs(A)
    if printer: print('newcoeff =')
    if printer: print(newcoeff, '\n')
        
    for i in range(N):
        c = newcoeff[:,i]
        if printer: print('c =')
        if printer: print(c, '\n')
        
        index = np.argsort(c)[::-1]
        if printer: print('index =')
        if printer: print(index, '\n')
        
        newcoeff[:,i] = newcoeff[:,i]/np.abs(c[index[0]])
        if printer: print('newcoeff =')
        if printer: print(newcoeff, '\n')
    
    if set_K != 0:
        newcoeff = newcoeff + newcoeff.T
        if printer: print('newcoeff =')
        if printer: print(newcoeff, '\n')
        if printer: print('k in not 0')
        
        index = np.argsort(newcoeff, axis=0)[::-1]
        if printer: print('index =')
        if printer: print(index, '\n')
        
        k_coeff = np.zeros((N, N))
        if printer: print('k_coeff =')
        if printer: print(k_coeff, '\n')
        
        for i in range(N):
            for j in range(K):
                k_coeff[index[j,i],i] = newcoeff[index[j,i],i]/newcoeff[index[0,i],i]
                if printer: print('k_coeff =')
                if printer: print(k_coeff, '\n')
        newcoeff = k_coeff
        
    if printer: print('end newcoeff =')
    if printer: print(newcoeff, '\n')
    
    return newcoeff