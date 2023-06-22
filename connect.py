import numpy as np
import label_check as lc
import normalize as no
import matplotlib.pyplot as plt

def connect(S, gt, L, kmeans_labels, nmi, prob_type, prob_title, printer=False, save_figure=False, sigma=0, gamma=0):
    '''
    plot connection strenghts
    lines are ground truth
    points are cluster results from kmeans
    keyword arguments:
        S: data points
        gt: ground truth
        L: normalized symmetric laplacian
        kmeans_labels: kmeans solution
        nmi: normalized mutual information score
        prob_type: name of clustering type used
        prob_title: name of the problem clustered
        printer: print out statments for debugging default=False
        save_figure: save figure to figures folder
    '''
    
    labels = lc.label_check(kmeans_labels, printer)
    
    if prob_type == 'spectral':
        text = 'figures/' + prob_type + '_connections_' + prob_title.replace("s/", '_').replace(".", '_') + '_' + str(sigma) + '.png'
        title = prob_type + '_' + prob_title + '\nnormalized mutual information score: ' + str(nmi) + ' sigma: ' + str(sigma)
        
    if prob_type == 'subspace':
        text = 'figures/' + prob_type + '_connections_' + prob_title.replace("s/", '_').replace(".", '_') + '_' + str(gamma) + '.png'
        title = prob_type + '_' + prob_title + '\nnormalized mutual information score: ' + str(nmi) + ' gamma: ' + str(gamma)
        
    if prob_type == 'hybrid':
        text = 'figures/' + prob_type + '_connections_' + prob_title.replace("s/", '_').replace(".", '_') + '_' + str(sigma) + '_' + str(gamma) + '.png'
        title = prob_type + '_' + prob_title + '\nnormalized mutual information score: ' + str(nmi) + ' sigma: ' + str(sigma) + 'gamma: ' + str(gamma)
    
    L = no.normalize(L, False)
    
    plt.scatter(S[:,0], S[:,1], color = [["red", "blue"][i] for i in labels])
    
    for i in range(L.shape[0]):
        for j in range(L.shape[0] - i):
            if i==j+i: continue
            if labels[i] == labels[j+i]:
            # if gt[i] == gt[j+i]:
            # if gt[i] == labels[i]: edit made
                plt.plot(S[i:j+i+1:j][:,0], S[i:j+i+1:j][:,1], color='green', alpha=L[i,j+i])
            else:
                plt.plot(S[i:j+i+1:j][:,0], S[i:j+i+1:j][:,1], color='purple', alpha=L[i,j+i])
    plt.title(title)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.savefig(text, bbox_inches="tight")
    plt.show()