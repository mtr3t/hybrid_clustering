import numpy as np

def load_problem(problem, ground_truth, printer=False, plotter=False, save_figure=False):
    '''
    load the problem to cluster
        problem: problem to cluster
        ground_truth: ground truth of the problem
        printer: print variables
        plotter: plot problem with ground truth
        save_figure: save figure to figures folder
    returns:
        Y: data points
        gt: ground truth
        n: number of clusters
        
    '''
    
    # Y = a set of y points = {y_1,...,y_n} in R^l
    Y = np.loadtxt(problem)
    if printer: print('problem:')
    if printer: print(Y, '\n')
    if printer: print('problem size:', Y.shape[0], 'x', Y.shape[1], '\n')
        
    # load the ground truth as gt  
    gt = np.loadtxt(ground_truth).astype(np.int32)
    if printer: print('len ground truth:', len(gt), '\n')
    if printer:
        print('ground truth:')
        print(gt[:, None], '\n')
    
    # calculate the nuber of clusters, n, from the ground truth
    n = len(np.unique(gt))
    if printer: print('number of clusters n:', n, '\n')
    
    if plotter:
        if Y.shape[1] == 2:
            import matplotlib.pyplot as plt
            
            plt.scatter(Y[:,0], Y[:,1], color = [["red", "blue"][i] for i in gt])
            plt.title(problem)
            plt.ylabel('y')
            plt.xlabel('x')
            if save_figure == True:
                text = 'figures/' + str(problem).replace("s/", '_').replace(".", '_') + '.png'
                plt.savefig(text)
            plt.show()
            
    return Y, gt, n