import numpy as np
import label_check as lc
import normalize as no
import matplotlib.pyplot as plt

def connect(S, gt, L, kmeans_labels, nmi, prob_type, prob_title, sigma, printer=False, save_figure=False):
    '''
    plot connection strenghts
    lines are fround truth
    points are cluster results from kmeans
    '''
    
    labels = lc.label_check(kmeans_labels, printer)
    
    text = 'figures/' + prob_type + '_connections_' + prob_title.replace("s/", '_').replace(".", '_') + '_' + str(sigma) + '.png'
    title = prob_type + '_' + prob_title + '\nnormalized mutual information score: ' + str(nmi) + ' sigma: ' + str(sigma)
    
    L = no.normalize(L, False)
    
    plt.scatter(S[:,0], S[:,1], color = [["red", "blue", "green"][i] for i in labels])
    
    for i in range(L.shape[0]):
        for j in range(L.shape[0] - i):
            if i==j+i: continue
            if gt[i] == gt[j+i]:
            # if gt[i] == labels[i]: edit made
                plt.plot(S[i:j+i+1:j][:,0], S[i:j+i+1:j][:,1], color='green', alpha=L[i,j+i])
            else:
                plt.plot(S[i:j+i+1:j][:,0], S[i:j+i+1:j][:,1], color='purple', alpha=L[i,j+i])
    plt.title(title)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.savefig(text, bbox_inches="tight")
    plt.show()