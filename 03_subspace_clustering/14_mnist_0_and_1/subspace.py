import sys
import time
import numpy as np
import load_problem as ld
import plotter as pl
import minimize as mn
import connect as ct
import row_normalization as rn
import symmetrize as sy
import adjacency as adj
import normalize as nr
import spectral as sp
import eigen_solver as es
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score

def subspace_clustering(set_gamma=0.001,
                        set_K=0,
                        norm_sub=False,
                        plotter=False,
                        extra_plotter=False,
                        printer=False,
                        save_figure=False,
                        axis_hold=False):
    """perform subspace clustering
    keyword arguments:
        problem: to cluster
        ground_truth: of the problem
        set_gamma: convex factor
        set_K: sets the number of K's <- better desc :(
        norm_sub: norilize matrix or not
        plotter: plot the problem default=False
        final_plotter : plot the mutual connections and final plot after k_means
        printer: print out statments for debugging default=False
    returns:
        normalized mutual information score
    """
    
    start = time.time()
    
    problem = '../../06_datasets/02_mnist/05_mnist_0_and_1_78.ds'
    ground_truth = '../../06_datasets/02_mnist/05_mnist_0_and_1_78_ground_truth.ds'
    
    S, gt, k = ld.load_problem(problem, ground_truth, printer, plotter, False, axis_hold)
            
    # begin subspace clustering
    
    coeff, A = mn.minimize(S, set_gamma, printer)
    
    if norm_sub: coeff = rn.row_normalizer(coeff, printer)

    coeff = sy.symmetrize(coeff, printer)    

    newcoeff = adj.adjacency(coeff, set_K, A, printer)

    if norm_sub: newcoeff = rn.row_normalizer(newcoeff, printer)
    
    newcoeff = sy.symmetrize(newcoeff, printer)
    
    # perform spectral clustering
    
    L = sp.subspace_spectral(newcoeff, printer, plotter)
    
    X, Y = es.eigen_solver(L, k, printer, extra_plotter)
    
    # perform kmeans
    kmeans = KMeans(n_clusters=k, max_iter=1000, n_init=20).fit(Y)
          
    # calculate the normalized mutual information score
    nmi = normalized_mutual_info_score(gt, kmeans.labels_)
    print(problem, '\ngamma:', round(set_gamma, 3), '\nk:', set_K, '\nnormalized mutual information score:', round(nmi, 3))

    end = time.time()
    print("time of execution :", (end-start) / 60, "min")
    
    if plotter:
        if S.shape[1] == 2:           
            prob_type = 'subspace'
            prob_title = str(problem)
            pl.plotter(S, kmeans.labels_, nmi, prob_type, prob_title, set_gamma, printer, save_figure, axis_hold)
            ct.connect(S, gt, L, kmeans.labels_, nmi, prob_type, prob_title, set_gamma, printer, save_figure, axis_hold)
            

if __name__ == '__main__':
    set_gamma = float(sys.argv[1]) #0.25
    set_K = float(sys.argv[2]) #0
    plotter = False
    printer = False
    norm_sub = False
    extra_plotter = False
    subspace_clustering(set_gamma, set_K, norm_sub, plotter, extra_plotter, printer)