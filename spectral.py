import time
import numpy as np
import load_problem as ld
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score

def spectral(problem, ground_truth, sigma):
    """perform spectral clustering
    keyword arguments:
        problem: to cluster
        ground_truth: known ground truth of the problem
    returns:
        prints normalized mutual information score
    """

    start = time.time()

	Y, gt, k = ld.load_problem(problem, ground_truth)

    s_dist = sdist.squareform(sdist.pdist(Y)) # pairwise distance
    # sigma = 1 # affinity scaler
    A_sp = np.exp((-1.0*np.power(s_dist,2))/(2.0 * np.power(sigma,2))) # affinity matrix
    A_sp[range(A_sp.shape[0]),range(A_sp.shape[1])] = 0.0 # remove diag

    D = np.diagflat(1.0/np.sqrt(np.sum(A_sp, axis=1))) # sum the rows (cols = 0 and rows = 1)
    L = D @ A_sp @ D
    e_vals, e_vecs = np.linalg.eigh(L)
    top_n_e_vecs = []
    for i in range(k):
    	top_n_e_vecs.append(-1 * (i+1))
    X = e_vecs[:,top_n_e_vecs]
    Y = np.divide(X, np.sqrt(np.sum(np.square(X), axis=1))[:, None])
    kmeans = KMeans(k, max_iter=1000, n_init=20).fit(Y)
    nmi = normalized_mutual_info_score(gt, kmeans.labels_)

    print('normalized mutual information score:', nmi, 'sigma', sigma)

    end = time.time()
    print("time of execution :", (end-start) / 60, "min")
    
    
if __name__ == '__main__':
    spectral('01_binary_alpha_digits_1404.ds', '01_binary_alpha_digits_1404_ground_truth.ds, 0.1)
