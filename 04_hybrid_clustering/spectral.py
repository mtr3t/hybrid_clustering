import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as sdist

def spectral(S, sigma, printer, plotter):
    '''
    performs spectral clustering
    returns:
        A: affinity matrix
        L: normalized symmetric laplacian
    '''
    
    if printer: print('begin spectral method', '\n')
    
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
    
    ###############################################################
    # get the row sums and calculate D**-1/2                      #
    # rows = 1, col = 0                                           #
    D = np.diagflat(1.0/np.sqrt(np.apply_along_axis(np.sum,1,A))) #
    if printer: print('diagonal matrix, D:')                      #
    if printer: print(D, '\n')                                    #
    ###############################################################
    
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
        
    if printer: print('end spectral method', '\n')
        
    return A, L

def subspace_spectral(A, printer, plotter):
    '''
    performs spectral clustering
    returns:
        L: normalized symitric laplacian
    '''
    
    if plotter: print('begin subspace_spectral method', '\n')
        
    ###############################################################
    # get the row sums and calculate D**-1/2                      #
    # rows = 1, col = 0                                           #
    D = np.diagflat(1.0/np.sqrt(np.apply_along_axis(np.sum,1,A))) #
    if printer: print('diagonal matrix, D:')                      #
    if printer: print(D, '\n')                                    #
    ###############################################################

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
    
    if printer: print('end subspace_spectral method', '\n')
    
    return L

def hybrid_spectral(A, printer, plotter):
    '''
    performs spectral clustering
    returns:
        L: normalized symitric laplacian
    '''
    
    if plotter: print('begin hybrid_spectral method', '\n')

    ###############################################################
    # get the row sums and calculate D**-1/2                      #
    # rows = 1, col = 0                                           #
    D = np.diagflat(1.0/np.sqrt(np.apply_along_axis(np.sum,1,A))) #
    if printer: print('diagonal matrix, D:')                      #
    if printer: print(D, '\n')                                    #
    ###############################################################

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
    
    if plotter: print('end hybrid_spectral method', '\n')
    
    return L

def hybrid_spectral_col(A, printer, plotter):
    '''
    performs spectral clustering
    returns:
        L: normalized symitric laplacian
    '''
    
    if plotter: print('begin hybrid_spectral_col method', '\n')
        
   ###############################################################
    # get the row sums and calculate D**-1/2                      #
    # rows = 1, col = 0                                           #
    D = np.diagflat(1.0/np.sqrt(np.apply_along_axis(np.sum,0,A))) #
    if printer: print('diagonal matrix, D:')                      #
    if printer: print(D, '\n')                                    #
    ###############################################################

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
    
    if plotter: print('end hybrid_spectral_col method', '\n')
    
    return L

def hybrid_spectral_din_col_dout_row(A, printer, plotter):
    '''
    performs spectral clustering
    returns:
        L: normalized symitric laplacian
    '''
    
    if plotter: print('begin hybrid_spectral_din_col_dout_row method', '\n')
    
    # rows = 1, col = 0 
    D_in = np.diagflat(1.0/np.sqrt(np.apply_along_axis(np.sum,0,A)))
    D_out = np.diagflat(1.0/np.sqrt(np.apply_along_axis(np.sum,1,A)))

    # plot
    if plotter:
        plt.imshow(D_in)
        plt.title('D_in_col' )
        plt.show()
        
    if plotter:
        plt.imshow(D_out)
        plt.title('D_out_row' )
        plt.show()

    # normilization
    L = D_out @ A @ D_in
    if printer: print('L matrix, L:')
    if printer: print(L, '\n')

    # plot
    if plotter:
        # plt.imshow(L, origin='lower')
        plt.imshow(L)
        plt.title('L' )
        plt.show()
    
    if plotter: print('end hybrid_spectral_din_col_dout_row method', '\n')
    
    return L

def hybrid_spectral_din_row_dout_col(A, printer, plotter):
    '''
    performs spectral clustering
    returns:
        L: normalized symitric laplacian
    '''
    
    if plotter: print('begin hybrid_spectral_din_row_dout_col method', '\n')
    
    # rows = 1, col = 0 
    D_in = np.diagflat(1.0/np.sqrt(np.apply_along_axis(np.sum,1,A)))
    D_out = np.diagflat(1.0/np.sqrt(np.apply_along_axis(np.sum,0,A)))

    # plot
    if plotter:
        plt.imshow(D_in)
        plt.title('D_in_col' )
        plt.show()
        
    if plotter:
        plt.imshow(D_out)
        plt.title('D_out_row' )
        plt.show()

    # normilization
    L = D_out @ A @ D_in
    if printer: print('L matrix, L:')
    if printer: print(L, '\n')

    # plot
    if plotter:
        # plt.imshow(L, origin='lower')
        plt.imshow(L)
        plt.title('L' )
        plt.show()
    
    if plotter: print('end hybrid_spectral_din_row_dout_col method', '\n')
    
    return L

def hybrid_spectral_din_dout_meila(A, printer, plotter):
    '''
    performs spectral clustering
    returns:
        L: normalized symitric laplacian
    '''
    
    if plotter: print('begin hybrid_spectral_din_dout_meila method', '\n')
        
    ones = np.ones(A.shape)
    D_in = (np.diag(ones.T@A))
    D_out = (np.diag(A@ones))
    D_sym = 0.5 * (D_in + D_out)

    # plot
    if plotter:
        plt.imshow(D_in)
        plt.title('D_in_col' )
        plt.show()
        
    if plotter:
        plt.imshow(D_out)
        plt.title('D_out_row' )
        plt.show()
    
    if plotter:
        plt.imshow(D_sym)
        plt.title('D_out_sym' )
        plt.show()

    # normilization
    L = np.diag(1/(np.sqrt(D_sym))) @ (0.5 * (A + A.T)) @ np.diag(1/(np.sqrt(D_sym)))
    if printer: print('L matrix, L:')
    if printer: print(L, '\n')

    # plot
    if plotter:
        # plt.imshow(L, origin='lower')
        plt.imshow(L)
        plt.title('L' )
        plt.show()
    
    if plotter: print('end hybrid_spectral_din_dout_meila method', '\n')
    
    return L

# last one to try is the hermitian matrix
# def hybrid_spectral_atev(A, printer, plotter):
#     '''
#     performs spectral clustering
#     returns:
#         L: normalized symitric laplacian
#     '''
    
#     if plotter: print('begin hybrid_spectral method', '\n')
        
# #     # this code is wrong
# #     ones = np.ones(A.shape)
# #     D_in = (np.diag(ones.T@A))
# #     D_out = (np.diag(A@ones))
# #     L = np.diag(1.0/np.sqrt((D_out))@ A@ np.diag(1.0/np.sqrt(D_in)))

#     ###############################################################
#     # get the row sums and calculate D**-1/2                      #
#     # rows = 1, col = 0                                           #
#     D = np.diagflat(1.0/np.sqrt(np.apply_along_axis(np.sum,1,A))) #
#     if printer: print('diagonal matrix, D:')                      #
#     if printer: print(D, '\n')                                    #
#     ###############################################################

# #     # plot
# #     if plotter:
# #         plt.imshow(D)
# #         # plt.imshow(D, origin='lower')
# #         plt.title('D' )
# #         plt.show()

#     # normilization
#     L = D @ A @ D
#     if printer: print('L matrix, L:')
#     if printer: print(L, '\n')

# #     D_in = np.diagflat(1.0/np.sqrt(np.apply_along_axis(np.sum,0,A)))
# #     D_out = np.diagflat(1.0/np.sqrt(np.apply_along_axis(np.sum,1,A)))
# #     L = D_out @ A @ D_in

#     # plot
#     if plotter:
#         # plt.imshow(L, origin='lower')
#         plt.imshow(L)
#         plt.title('L' )
#         plt.show()
    
#     if plotter: print('end hybrid_spectral method', '\n')
    
#     return L

