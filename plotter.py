import label_check as lc
import matplotlib.pyplot as plt

def plotter(S, kmeans_labels, nmi,  prob_type, prob_title, printer=False, save_figure=False):
    '''
    plots the labels as chosen by kmeans
    '''

    # flip the bits if needed
    labels = lc.label_check(kmeans_labels, printer)
    
    if printer: print('k-means labels are:')
    if printer: print(labels[:, None], '\n')
    
    text = 'figures/' + prob_type + '_solve_' + prob_title.replace("s/", '_').replace(".", '_') + '.png'
    title = prob_type + '_' + prob_title + '\nnormalized mutual information score: ' + str(nmi)

    # print out problem
    plt.scatter(S[:,0], S[:,1], color = [["red", "blue"][i] for i in labels])
    plt.title(title)
    plt.ylabel('y')
    plt.xlabel('x')
    if save_figure: plt.savefig(text)
    plt.show()