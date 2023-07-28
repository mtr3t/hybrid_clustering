import numpy as np
import matplotlib.pyplot as plt

def eigen_solver(L, k, printer=False, plotter=False):
    '''
    solves for the eigenvalues and eigenvectors
    input:
        L: normalized symmetric laplacian
        k: number of clusters
        printer: prints all variables
        plotter: plots all variables
    returns:
        X: top egienvectors
        Y: X normalized
    '''
    
    # calculate eigenvalues and eigenvectors
    e_vals, e_vecs = np.linalg.eigh(L)

    if printer: print('eigenvalues:')
    if printer: print(e_vals, '\n')
    if printer: print('eigenvectors:')
    if printer: print(e_vecs, '\n')

    if plotter:
        plt.figure(figsize=(5, 5))
        plt.scatter(np.linspace(1,len(e_vals), num=len(e_vals)), e_vals)
        plt.title("eigenvalues" )
        plt.show()

    top_k_e_vecs = []
    for i in range(k):
        top_k_e_vecs.append(-1 * (i+1))
    
    if printer: print('top k egienvectors:')
    if printer: print(top_k_e_vecs, '\n')
        
    X = e_vecs[:,top_k_e_vecs]
    if printer: print('top k egienvectors stacked in columns, X:')
    if printer: print(X, '\n')
    
    X_squared = np.square(X)
    if printer: print('X squared:')
    if printer: print(np.square(X), '\n')
        
    row_sum = np.apply_along_axis(np.sum,1,X_squared)
    if printer: print('X squared row sum:')
    if printer: print(row_sum[:, None], '\n')
        
    # create a boolean mask of elements equal to 0
    mask = (row_sum == 0)
    
    if np.any(mask):
        print("0 is in the array\n")
        row_sum[mask] = 1.0
    
#     Y = np.zeros(X.shape)
#     for i in range(X.shape[0]):
#         for j in range(X.shape[1]):
#             Y[i,j] = X[i,j] * (1/np.sqrt(row_sum[j]))

#     if printer: print('renormalized matrix, Y:')
#     if printer: print(Y, '\n')
        
    norm_row_sum = np.sqrt(row_sum)
    if printer: print(norm_row_sum[:, None], '\n')

    Y = np.divide(X, norm_row_sum[:, None])
    if printer: print('renormalized matrix, Y:')
    if printer: print(Y, '\n')


    if plotter:
        if Y.shape[1] == 2:
            plt.figure(figsize=(5, 5))
            plt.scatter(Y[:,0], Y[:,1])
            plt.ylabel('y')
            plt.xlabel('x')
            plt.title('Y, points are tightly grouped together')
            plt.show()
            
    return X, Y