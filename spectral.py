import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as sdist

def spectral_affinity(S, sigma, printer=False, plotter=False):
    '''
    calculates spectral affinity matrix
    returns:
        A: spectral affinity matrix
    '''
    
    if printer: print('begin spectral affinity', '\n')
    
    # calculate pairwise distance (euclidean)
    s_dist = sdist.squareform(sdist.pdist(S))
    if printer: print('pairwise distance (euclidean):')
    if printer: print(s_dist, '\n')

    # calculate the affinity matrix A
    # sigma is the affinity scaling factor
    # 0 to infinity spread of gaussian
    A = np.exp((-1.0*np.power(s_dist,2))/(2.0 * np.power(sigma,2)))
    if printer: print("affinity matrix with 1's on diag:")
    if printer: print(A, '\n')
    
    # plot affinity matrix
    if plotter:
        # plt.imshow(A, origin='lower')
        plt.imshow(A)
        plt.title("affinity matrix with 1's on diag" )
        plt.show()

    # remove the ones on the diag
    A[range(A.shape[0]),range(A.shape[1])] = 0.0
    if printer: print('affinity matrix complete, A:')
    if printer: print(A, '\n')
    
    # plot
    if plotter:
        # plt.imshow(A, origin='lower')
        plt.imshow(A)
        plt.title("affinity matrix with 1's removed on diag" )
        plt.show()
        
    if printer: print('end spectral affinity', '\n')
        
    return A

def laplacian(A, printer, plotter):
    '''
    performs normalized symitric laplacian on incoming affinty matrix
    returns:
        L: normalized symitric laplacian
    '''
    
    if printer: print('begin normalized symitric laplacian', '\n')
        
    # get the row sums and calculate D**-1/2
    # rows = 1, col = 0
    D = np.diagflat(1.0/np.sqrt(np.apply_along_axis(np.sum,1,A)))
    if printer: print('diagonal matrix, D:')
    if printer: print(D, '\n')

    # plot
    if plotter:
        plt.imshow(D)
        # plt.imshow(D, origin='lower')
        plt.title('D' )
        plt.show()

    # normilization
    L = D @ A @ D
    if printer: print('L matrix, L:')
    if printer: print(L, '\n')

    # plot
    if plotter:
        # plt.imshow(L, origin='lower')
        plt.imshow(L)
        plt.title('L' )
        plt.show()
    
    if printer: print('end normalized symitric laplacian', '\n')
    
    return L