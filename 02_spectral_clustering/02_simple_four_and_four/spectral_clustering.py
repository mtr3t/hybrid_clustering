import sys
import time
import numpy as np
import load_problem as ld
import spectral as sp
import eigen_solver as es
import plotter as pl
import connect as ct
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score

def spectral_clustering(sigma=0.25,
                        printer=False,
                        plotter=False,
                        extra_plotter=False,
                        save=False,
                        axis_hold=False):
    """perform spectral clustering acording to ng, jordan and weiss
    keyword arguments:
        problem: to cluster
        ground_truth: of the problem
        sigma: affinity scaling factor
        printer: print out statments for debugging default=False
        plotter: plot the problem default=False
        extra_plotter : plot the mutual connections and final plot after k_means
   s     save: saves figures to 07_figures in respective folders
    variables:
        S: data points
        gt: ground truth
        k: number of clusters
        A: affinity matrix
        L: normalized symmetric laplacian
        X: top k egienvectors stacked in columns
        Y: renormalized X matrix
    returns:
        normalized mutual information score
    """
    
    start = time.time()
    
    problem = '../../05_toy_problems/02_simple_four_and_four.tp'
    ground_truth = '../../05_toy_problems/02_simple_four_and_four_ground_truth.tp'
    
    # load the problem
    S, gt, k = ld.load_problem(problem, ground_truth, printer, plotter, False, axis_hold)
    
    A, L = sp.spectral(S, sigma, printer, extra_plotter)
    
    X, Y = es.eigen_solver(L, k, printer, extra_plotter)
    
    # perform kmeans
    kmeans = KMeans(n_clusters=k, max_iter=1000, n_init=20).fit(Y)
          
    # calculate the normalized mutual information score
    nmi = normalized_mutual_info_score(gt, kmeans.labels_)
    print('problem :', problem, '\nsigma:', round(sigma, 4),
          '\nnormalized mutual information score:', round(nmi, 4))
    
    end = time.time()
    print("time of execution :", (end-start) / 60, "min")
    
    if plotter:
        if S.shape[1] == 2:           
            prob_type = 'spectral'
            prob_title = str(problem)
            pl.plotter(S, kmeans.labels_, nmi, prob_type, prob_title, sigma, printer, save, axis_hold)
            ct.connect(S, gt, L, kmeans.labels_, nmi, prob_type, prob_title, sigma, printer, save, axis_hold)

if __name__ == '__main__':
    sigma = float(sys.argv[1]) #0.25
    printer = False
    plotter = False
    extra_plotter = False
    save = False
    axis_hold = False
    spectral_clustering(sigma, printer, plotter, extra_plotter, save, axis_hold)