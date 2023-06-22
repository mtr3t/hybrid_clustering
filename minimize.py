import numpy as np
import cvxpy as cp

def minimize(S, set_gamma, printer=False):
    """minimize S using cvxpy
    keyword arguments:
        S: data poins
        set_gamma: convex factor
        printer: print out statments for debugging default=False
    returns:
        A: affinity minimized
    """
    
    # transpose S 
    S_transpose = np.transpose(S)
    if printer: print('S_transpose =')
    if printer: print(S_transpose, '\n')
    if printer: print('shape of S_transpose =', np.shape(S_transpose), '\n')
        
    # this is where you can perform a data projection
    # this is skipped at the moment
    # possible projections PCA, NormalProj, BernoulliProj
        
    # create coefficent matrix A
    A = np.zeros([np.shape(S_transpose)[1],np.shape(S_transpose)[1]])
    if printer: print('create A =')
    if printer: print(A, '\n')
    
    if printer: print('\n************************************************' +
                      '\n            Solve for coefficents')    
    
    for i in range(np.shape(A)[1]):
        b = S_transpose[:,i]
        if printer: print('b =\n', b)

        # gamma must be positive due to DCP rules.
        gamma = cp.Parameter(nonneg="true") 
        gamma.value = set_gamma
        
        # Construct the problem.
        x = cp.Variable(np.shape(S_transpose)[1])
        
        # Lasso
        obj = cp.Minimize(gamma*cp.norm(S_transpose@x-b,2) + cp.norm(x,1))
        constraint = x[i] == 0
 
        # # L1-Perfect
        # obj = Minimize(norm(x, 1))
        # constraint = [A*x == b, x[i] == 0, sum(x) == 1]
        
        # # L1-Noisy
        # obj = Minimize(norm(x, 1))
        # constraint = [ A*x - b <= gamma, x[i] == 0, sum(x) == 1 ]

        # if constraint == None:
            # prob = cp.Problem(obj)
        # else:
        
        prob = cp.Problem(obj, [constraint])
    
        prob.solve(solver='ECOS')

        A[:,i] = np.transpose(x.value)
        if printer: print('A =')
        if printer: print(A,'\n')
            
    if printer: print('\n************************************************' +
                      '\n         Done solving for coefficents\n')
    # zero the diag
    A[range(A.shape[0]),range(A.shape[1])] = 0.0
    if printer: print('A =')
    if printer: print(A,'\n')

    ## Refine results...
    ## Only use magnitude of the coefficients (no negative values)
    A = np.abs(A)
    if printer: print('abs of A =')
    if printer: print(A, '\n')
    
    return A