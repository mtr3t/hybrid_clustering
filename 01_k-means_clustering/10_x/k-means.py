import numpy as np
import plotter as pl
import load_problem as ld
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score

def k_means(printer=False, plotter=False, save_figure=False, axis_hold=False):
    """perform kmeans clustering
    keyword arguments:
        problem: to cluster
        ground_truth: known ground truth of the problem
        printer: print out statments for debugging default=False
        plotter: plot the problem default=False
        save_figure: save figure to figures folder
    returns:
        prints problem location
        prints normalized mutual information score
    """
    # load the problem
    # returns:
        # Y: data points
        # gt: ground truth
        # k: number of clusters for k_means
        
    problem = '../../05_toy_problems/10_x.tp'
    ground_truth = '../../05_toy_problems/03_ground_truth.tp'
    
    Y, gt, k = ld.load_problem(problem, ground_truth, printer, plotter, save_figure, axis_hold)
        
    # perform kmeans
    kmeans = KMeans(n_clusters=k, max_iter=1000, n_init=20).fit(Y)
          
    # calculate the normalized mutual information score
    nmi = normalized_mutual_info_score(gt, kmeans.labels_)
    print(problem, '\nnormalized mutual information score:', nmi)
    
    if plotter:
        if Y.shape[1] == 2:           
            prob_type = 'k-means'
            prob_title = str(problem)
            pl.plotter(Y, kmeans.labels_, nmi, prob_type, prob_title, printer, save_figure, axis_hold)

if __name__ == '__main__':
    printer = False
    plotter = False
    save_figure = False
    axis_hold = False
    k_means(printer, plotter, save_figure, axis_hold)