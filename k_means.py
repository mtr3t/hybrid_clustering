import time
import numpy as np
import load_problem as ld
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score

def k_means(problem, ground_truth):
    """perform kmeans clustering
    keyword arguments:
        problem: to cluster
        ground_truth: known ground truth of the problem
    returns:
        prints normalized mutual information score
    """
    # load the problem
    # returns:
        # Y: data points
        # gt: ground truth
        # k: number of clusters for k_means
    
    start = time.time()
    
    Y, gt, k = ld.load_problem(problem, ground_truth)
        
    # perform kmeans
    kmeans = KMeans(n_clusters=k, max_iter=1000, n_init=20).fit(Y)
          
    # calculate the normalized mutual information score
    nmi = normalized_mutual_info_score(gt, kmeans.labels_)
    print(problem, '\nnormalized mutual information score:', nmi)
    
    end = time.time()
    print("time of execution :", (end-start) / 60, "min")
    
    
if __name__ == '__main__':
    k_means('01_binary_alpha_digits_1404.ds', '01_binary_alpha_digits_1404_ground_truth.ds)