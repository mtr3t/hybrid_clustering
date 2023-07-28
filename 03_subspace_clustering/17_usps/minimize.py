import numpy as np
import cvxpy as cp

def minimize(S, set_gamma, printer=False, set_constraint=1):
    
    # transpose S 
    A = np.transpose(S)
    if printer: print('A =')
    if printer: print(A, '\n')
    if printer: print('shape of A =', np.shape(A), '\n')
        
    # this is where you can perform a data projection
    # this is skipped at the moment
    # possible projections PCA, NormalProj, BernoulliProj
        
    # create coefficent matrix
    coeff = np.zeros([np.shape(A)[1],np.shape(A)[1]])
    if printer: print('coeff =')
    if printer: print(coeff, '\n')
    
    if printer: print('\n************************************************' +
                      '\n            Solve for coefficents')    
    
    for i in range(np.shape(A)[1]):
        b = A[:,i]
        if printer: print('b =\n', b)

        # gamma must be positive due to DCP rules.
        gamma = cp.Parameter(nonneg="true") 
        gamma.value = set_gamma
        
        # Construct the problem.
        x = cp.Variable(np.shape(A)[1])
        
        if set_constraint == 0:
            constraint = None
            # print('con none')
        else:
            constraint = x[i] == 0
            # print('con zeros')

        # Lasso
        obj = cp.Minimize(gamma*cp.norm(A@x-b,2) + cp.norm(x,1))
        # obj = cp.Minimize(set_gamma*cp.norm(A@x-b,2) + cp.norm(x, 1))
        # constraints = [x[i] == 0, sum(x) == 1]

        # # L1-Perfect
        # obj = Minimize(norm(x, 1))
        # constraints = [A*x == b, x[i] == 0, sum(x) == 1]
        
        # # L1-Noisy
        # obj = Minimize(norm(x, 1))
        # constraints = [ A*x - b <= gamma, x[i] == 0, sum(x) == 1 ]

        if constraint == None:
            prob = cp.Problem(obj)
        else:
            prob = cp.Problem(obj, [constraint])
    
        prob.solve(solver='ECOS')

        coeff[:,i] = np.transpose(x.value)
        if printer: print('coeff =')
        if printer: print(coeff,'\n')
            
    if printer: print('\n************************************************' +
                      '\n         Done solving for coefficents\n')
    # zero the diag
    coeff[range(coeff.shape[0]),range(coeff.shape[1])] = 0.0
    if printer: print('coeff =')
    if printer: print(coeff,'\n')

    ## Refine results...
    ## Only use magnitude of the coefficients (no negative values)
    coeff = np.abs(coeff)
    if printer: print('abs of coeff =')
    if printer: print(coeff, '\n')
    
    return coeff, A

