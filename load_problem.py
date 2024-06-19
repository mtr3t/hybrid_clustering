import numpy as np

def load_problem(problem, ground_truth):
    '''
    load the problem to cluster
        problem: problem to cluster
        ground_truth: ground truth of the problem
    returns:
        Y: data points
        gt: ground truth
        n: number of clusters
        
    '''
    
    # Y = a set of y points = {y_1,...,y_n} in R^l
    Y = np.loadtxt(problem)
        
    # load the ground truth as gt  
    gt = np.loadtxt(ground_truth).astype(np.int32)
    
    # calculate the nuber of clusters, n, from the ground truth
    k = len(np.unique(gt))
            
    return Y, gt, k