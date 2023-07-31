import sys
import time
import numpy as np
import load_problem as ld
import minimize as mn
import plotter as pl
import connect as ct
import symmetrize as sy
import adjacency as adj
import normalize as nr
import spectral as sp
import eigen_solver as es
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score

def hybrid(sigma=0.25, set_gamma=0.001,
           set_K=0,
           plotter=False,
           printer=False,
           save_figure=False,
           axis_hold=False):
    '''perform hybrid clustering
    keyword arguments:
        problem: to cluster
        ground_truth: of the problem
        set_gamma: convex factor
        set_K: sets the number of K's <- beeter desc
        norm_sub: norilize matrix or not
        plotter: plot the problem default=False
        final_plotter : plot the mutual connections and final plot after k_means
        printer: print out statments for debugging default=False
    returns:
        normalized mutual information score
    '''
    
    start = time.time()
    
    problem = '../../06_datasets/02_mnist/06_mnist_0_and_8_78.ds'
    ground_truth = '../../06_datasets/02_mnist/06_mnist_0_and_8_78_ground_truth.ds'
    
    # load the problem
    S, gt, k = ld.load_problem(problem, ground_truth, printer, plotter, False, axis_hold)
    
    # subspace
    A_min, A = mn.minimize(S, set_gamma, printer)
    
    A_sym = sy.symmetrize(A_min, printer)
    
        ### check what happens here ###
    A_ind = adj.adjacency(A_sym, set_K, printer)
    
    A_sub = sy.symmetrize(A_ind, printer)
    
    L_sub = sp.subspace_spectral(A_sub, printer, plotter)

    # spectral
    A_spc, L_spc = sp.spectral(S, sigma, printer, plotter)
    
    # hybrid
    # A_hyb = A_sub * A_spc
    A_hyb = np.matmul(A_sub, A_spc)
    # A_hyb = np.matmul(L_sub, L_spc)
    if printer: print('A_hyb: ')
    if printer: print(A_hyb)   

    ### stopped here on row/col sum 0 or 1??? ###
    L_hyb = sp.subspace_spectral(A_hyb, printer, plotter)
    
    X, Y = es.eigen_solver(L_hyb, k, printer, plotter)
    
    # perform kmeans
    kmeans = KMeans(n_clusters=k, max_iter=1000, n_init=20).fit(Y)
    
    # calculate the normalized mutual information score
    nmi = normalized_mutual_info_score(gt, kmeans.labels_)
    print(problem, '\nsigma:', round(sigma, 3), '\ngamma:', round(set_gamma, 3), '\nk:', set_K, '\nnormalized mutual information score:', round(nmi, 3))
    
    end = time.time()
    print("time of execution :", (end-start) / 60, "min")
    
    if plotter:
        if S.shape[1] == 2:           
            prob_type = 'hybrid'
            prob_title = str(problem)
            sigma_gamma = [sigma, set_gamma]
            pl.plotter(S, kmeans.labels_, nmi, prob_type, prob_title, sigma_gamma, printer, save_figure, axis_hold)
            ct.connect(S, gt, L_hyb, kmeans.labels_, nmi, prob_type, prob_title, sigma_gamma, printer, save_figure, axis_hold)
            
if __name__ == '__main__':
    sigma = float(sys.argv[1]) #0.25
    set_gamma = float(sys.argv[2]) #0.1
    set_K = int(sys.argv[3]) #0
    plotter = False
    printer = False
    save_figure = False
    hybrid(sigma, set_gamma, set_K, plotter, printer, save_figure)