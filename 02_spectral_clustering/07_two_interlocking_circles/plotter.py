import label_check as lc
import matplotlib.pyplot as plt

def plotter(Y, kmeans_labels, nmi, prob_type, prob_title, sigma, printer=False, save_figure=False, axis_hold=False):
    '''
    plots the labels as chosen by kmeans
        Y: data points
        kmeans_labels: kmeans solution
        nmi: normalized mutual information score
        prob_type: name of clustering type used
        prob_title: name of the problem clustered
        printer: print out statments for debugging default=False
        save_figure: save figure to figures folder
        axis_hold: for toy probles that need the same axis to show different distances
        
    '''

    # flip the bits if needed
    labels = lc.label_check(kmeans_labels, printer)
    
    if printer: print('k-means labels are:')
    if printer: print(labels[:, None], '\n')
    
    text = '../07_figures/02_spectral_clustering/' + prob_type + '_solve_' + prob_title.replace("../05_", '').replace("s/", '_').replace(".", '_') + '_sigma_' + str(sigma).replace(".", '_') + '_nmi_' + str(nmi).replace(".", '_') + '.png'
    # title = prob_type + '_' + prob_title + '\nnormalized mutual information score: ' + str(nmi)

    # print out problem
    
    plt.figure(figsize=(5, 5))
    
    plt.scatter(Y[:,0], Y[:,1], color = [["red", "blue"][i] for i in labels])
    # plt.title(title)
    plt.ylabel('y')
    plt.xlabel('x')
    if axis_hold:
        plt.xlim(-1, 11)
        plt.ylim(-2, 8)
    # plt.axis('equal')
    if save_figure: plt.savefig(text)
    plt.show()