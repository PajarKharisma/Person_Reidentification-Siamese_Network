import numpy as np
import matplotlib.pyplot as plt

def imshow(img, text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_plot(xdata, y_data, title='', xlabel='', ylabel='', legend='', legend_loc='', path='', should_show='True', should_save='False'):
    for y in y_data:
        plt.plot(x_data, y)
        
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(legend, loc=legend_loc)
    if should_show:
        plt.show()
    if should_save:
        plt.savefig(path)
    plt.close()

def show_plot_roc(y_data, x_data, title='', xlabel='', ylabel=''):
    plt.plot(fpr, tpr, linestyle='--', label='Auc score : {}'.format(auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
