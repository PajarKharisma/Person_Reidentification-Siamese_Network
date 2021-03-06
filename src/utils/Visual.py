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

def show_plot(**kwargs):
    data = kwargs
    if data['type'] == 'val':
        plt.plot(data['epoch'], data['train_data'])
        plt.plot(data['epoch'], data['val_data'])
        plt.title(data['title'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['train', 'val'], loc='upper right')

    elif data['type'] == 'roc':
        plt.plot(data['fpr'], data['tpr'],  marker='.', label='Best Thresold : {}'.format(data['best_thresh']))
        plt.scatter(data['fpr'][data['ix']], data['tpr'][data['ix']], marker='o', color='black', label='Best')
        plt.title(data['title'])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        
    if data['should_show']:
        plt.show()
    if data['should_save']:
        plt.savefig(data['path'])
    plt.close()