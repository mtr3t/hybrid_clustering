import sys
import time
import numpy as np
import cvxpy as cp
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score

def subspace(problem, ground_truth, set_gamma, ntopco):
	"""perform subspace clustering
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

	A_sub = np.transpose(Y)
	coeff = np.zeros([np.shape(A_sub)[1],np.shape(A_sub)[1]])

	for i in range(np.shape(A_sub)[1]):
		b = A_sub[:,i]
		gamma = cp.Parameter(nonneg="true")
		gamma.value = set_gamma
		x = cp.Variable(np.shape(A_sub)[1])
		# constraint = x[i] == 0
		obj = cp.Minimize(gamma*cp.norm(A_sub@x-b,2) + cp.norm(x,1)) # Lasso
		prob = cp.Problem(obj) #, [constraint])
		prob.solve(solver='ECOS')
		coeff[:,i] = np.transpose(x.value)

	coeff[range(coeff.shape[0]),range(coeff.shape[1])] = 0.0

	coeff = np.abs(coeff)
	# coeff = coeff / np.max(coeff,axis=0)
	coeff = (coeff + np.transpose(coeff))

	# new code
	cabs = np.abs(coeff)
	for i in range(np.shape(cabs)[1]):
		c = np.abs(cabs[:,i])
		ind = np.argsort(c, axis=0)
		csorted = np.take_along_axis(c, ind, axis=0)
		under = csorted[-1]
		coeff[:,i] = coeff[:,i]/under
	newcoeff = coeff + coeff.T

	if ntopco > 0:
		newtopco = np.zeros(np.shape(coeff))
		ind = np.argsort(newcoeff, axis=0)
		for i in range(np.shape(newtopco)[1]):
			for j in range(ntopco):
				newtopco[ind[-1 * (j+1)][i]][i] = newcoeff[ind[-1 * (j+1)][i]][i] / newcoeff[ind[-1][i]][i]
		newcoeff = newtopco + newtopco.T

	D = np.diagflat(1.0/np.sqrt(np.sum(newcoeff, axis=1))) # sum the rows (cols = 0 and rows = 1)
	L = D @ newcoeff @ D
	e_vals, e_vecs = np.linalg.eigh(L)
	top_n_e_vecs = []
	for i in range(k):
		top_n_e_vecs.append(-1 * (i+1))
	X = e_vecs[:,top_n_e_vecs]
	Y = np.divide(X, np.sqrt(np.sum(np.square(X), axis=1))[:, None])
	kmeans = KMeans(k, max_iter=1000, n_init=20).fit(Y)
	nmi = normalized_mutual_info_score(gt, kmeans.labels_)

	print('normalized mutual information score:', nmi, 'gamma', set_gamma, ntopco)

	end = time.time()
	print("time of execution :", (end-start) / 60, "min")

if __name__ == '__main__':
    subspace('prob.tp', 'gt.tp', float(sys.argv[1]), float(sys.argv[2]))
