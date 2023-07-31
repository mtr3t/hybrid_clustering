import numpy as np
import label_check as lc
import normalize as no
import matplotlib.pyplot as plt

def connect(S, gt, L, kmeans_labels, nmi, prob_type, prob_title, sigma, printer=False, save_figure=False, axis_hold=False):
    '''
    plots the connection strenghts
        S: data points
        gt: ground truth of the problem
        kmeans_labels: kmeans solution
        nmi: normalized mutual information score
        prob_type: name of clustering type used
        prob_title: name of the problem clustered
        sigma: sigma chosen
        printer: print out statments for debugging default=False
        save_figure: save figure to figures folder
        axis_hold: for toy probles that need the same axis to show different distances
        
    '''
    
    labels = lc.label_check(kmeans_labels, printer)
    
    text = text = '../07_figures/04_hybrid_clustering/' + prob_type + '_connections_' + prob_title.replace("../05_", '').replace("s/", '_').replace(".", '_') + '_sigma_gamma_' + str(sigma).replace(".", '_').replace(", ", '_') + '_nmi_' + str(nmi).replace(".", '_') + '.png'
    # title = prob_type + '_' + prob_title + '\nnormalized mutual information score: ' + str(nmi)
    
    L = no.normalize(L, False)
    
    plt.figure(figsize=(5, 5))
    
    # this plots ground truth
    plt.scatter(S[:,0], S[:,1], color = [["red", "blue", "green"][i] for i in gt])
    
    # this plots k_means solution
    # plt.scatter(Y[:,0], Y[:,1], color = [["red", "blue"][i] for i in kmeans_labels])
    
    for i in range(L.shape[0]):
        for j in range(L.shape[0] - i):
            if i==j+i: continue
            # if gt[i] == gt[j+i]:
            if kmeans_labels[i] == kmeans_labels[j+i]:
            # if gt[i] == labels[i]:
                plt.plot(S[i:j+i+1:j][:,0], S[i:j+i+1:j][:,1], color='green', alpha=L[i,j+i])
            else:
                plt.plot(S[i:j+i+1:j][:,0], S[i:j+i+1:j][:,1], color='purple', alpha=L[i,j+i])
    # plt.title('spectral_connections_' + prob_title + '\nsigma_' + str(sigma))
    plt.ylabel('y')
    plt.xlabel('x')
    if axis_hold:
        plt.xlim(-1, 11)
        plt.ylim(-2, 8)
    if save_figure: plt.savefig(text)
    plt.show()