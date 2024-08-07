import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score

def k_means(problem, ground_truth):
    """perform kmeans clustering
    keyword arguments:
        problem: problem to cluster
        ground_truth: known ground truth of the problem
    returns:
        prints normalized mutual information score
    """
    
    start = time.time()
    
    # load problem
    Y = np.loadtxt(problem) # Y = a set of y points = {y_1,...,y_n} in R^l
    gt = np.loadtxt(ground_truth).astype(np.int32) # load the ground truth as gt
    k = len(np.unique(gt))    # calculate the nuber of clusters, n, from the ground truth
        
    # perform kmeans
    kmeans = KMeans(n_clusters=k, max_iter=1000, n_init=20).fit(Y)
          
    # calculate the normalized mutual information score
    nmi = normalized_mutual_info_score(gt, kmeans.labels_)
    print('k_means normalized mutual information score:', nmi)
    
    end = time.time()
    print("time of execution :", (end-start) / 60, "min")
    
    
if __name__ == '__main__':
    k_means('prob.tp', 'gt.tp')
