import numpy as np

def load_problem(problem, ground_truth, printer=False, plotter=False, save_figure=False):
    '''
    load the problem to cluster
        problem: problem to cluster
        ground truth: ground truth of the problem
        printer: print variables
        plotter: plot problem with ground truth
        save_figure: save figure to figures folder
    returns:
        S: data points
        gt: ground truth
        k:  number of clusters
        
    '''
    
    # S = a set of points = {s_1,...,s_n} in R^l
    S = np.loadtxt(problem)
    if printer: print('problem:')
    if printer: print(S, '\n')
    if printer: print('problem size:', S.shape[0], 'x', S.shape[1], '\n')
        
    # load the ground truth as gt  
    gt = np.loadtxt(ground_truth).astype(np.int32)
    if printer: print('len ground truth:', len(gt), '\n')
    if printer:
        print('ground truth:')
        print(gt[:, None], '\n')
    
    # calculate the nuber of clusters, k, from the ground truth
    k = len(np.unique(gt))
    if printer: print('number of clusters k:', k, '\n')
    
    if plotter:
        if S.shape[1] == 2:
            import matplotlib.pyplot as plt
            
            plt.scatter(S[:,0], S[:,1], color = [["red", "blue", "green"][i] for i in gt])
            plt.title(problem)
            plt.ylabel('y')
            plt.xlabel('x')
            if save_figure == True:
                text = 'figures/' + str(problem).replace("s/", '_').replace(".", '_') + '.png'
                plt.savefig(text)
            plt.show()
            
    return S, gt, k