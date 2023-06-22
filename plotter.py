import label_check as lc
import matplotlib.pyplot as plt

def plotter(Y, kmeans_labels, nmi, prob_type, prob_title, printer=False, save_figure=False):
    '''
    plots the labels as chosen by kmeans
        Y: data points
        kmeans_labels: kmeans solution
        nmi: normalized mutual information score
        prob_type: name of clustering type used
        prob_title: name of the problem clustered
        printer: print out statments for debugging default=False
        save_figure: save figure to figures folder
        
    '''

    # flip the bits if needed
    labels = lc.label_check(kmeans_labels, printer)
    
    if printer: print('k-means labels are:')
    if printer: print(labels[:, None], '\n')
    
    text = 'figures/' + prob_type + '_solve_' + prob_title.replace("s/", '_').replace(".", '_') + '.png'
    title = prob_type + '_' + prob_title + '\nnormalized mutual information score: ' + str(nmi)

    # print out problem
    plt.scatter(Y[:,0], Y[:,1], color = [["red", "blue"][i] for i in labels])
    plt.title(title)
    plt.ylabel('y')
    plt.xlabel('x')
    if save_figure: plt.savefig(text)
    plt.show()