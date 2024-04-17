"""
2022-02-03 MY Confusion matrix
2023-12-15 Modified for article experiment 1; best MLP for fusion and mosaics

python 19-confusionMatrices-experiment1.py

"""
import pandas as pd
from glob import glob

from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as colors
import pickle


cmap = plt.cm.Blues(np.linspace(0,1,20))
cmap = colors.ListedColormap(cmap[1:15,:-1])


outputpath = '/Users/myliheik/Documents/myVEGCOVER/vegcover2023/results/experiment1/img/confusionMatrices'

# EDIT input:
# Fused MLP:
#filepaths1 = glob('/Users/myliheik/Documents/myVEGCOVER/vegcover2023/results/experiment1/fusedS1S2/xtrain-ard-S1-S2-2D-noDuplicates_2020-2021-2022-2023-123450-Fusion-predictions-MLP-2-hidden_layers-200-iterations-0.001-alpha-l1-Fusion-123450-2024*')
# Mosaic MLP:
filepaths1 = glob('/Users/myliheik/Documents/myVEGCOVER/vegcover2023/results/experiment1/mosaicS1S2/xtrain-ard-S1-S2ind-2D-123450-Mosaics-predictions-MLP-2-hidden_layers-200-iterations-0.001-alpha-l1-Mosaic-123450-2024*')

filepaths = filepaths1
print(filepaths)

def plot_confusion_matrix(y_true, y_pred, trueLabel, predLabel, classes,
                          normalize=False,
                          title=None,
                          cmap=cmap,
                          filename=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = None
            #title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(classes, classes)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix, saved to:")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    #sns.set_style("whitegrid")
    #fig = plt.figure(figsize = (6, 2.5))
    #fig, ax = plt.figure(figsize = (2.25, 2.25))


    fig, ax = plt.subplots() 

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    #ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel=trueLabel,
           xlabel=predLabel)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
            
    plt.rcParams.update({'font.size': 11})
    fig.tight_layout()



    #np.set_printoptions(precision=2)


    if filename:
        print(filename)
        #plt.gcf().set_size_inches(3, 3)
        plt.savefig(filename, dpi=300)
        
    return ax



for file in filepaths:
    print(file)
    # parsing filename:
    if 'fused' in file.split('/')[-2]:
        timeMethod = 'fusion'
    elif 'timeseries' in file.split('/')[-2]:
        timeMethod = 'timeseries'
    else:
        timeMethod = 'mosaics'
        
    filename = file.split('/')[-1]
    loppu = filename.split('-')[-3:]
    
    #timeMethod = loppu[-1].split('.')[0]
    #classnr = filename.split('predictions')[1].split('-')[4]
    classifiername = filename.split('predictions')[1].split('-')[1]
    #print(timeMethod, classifiername, classnr)
    
    leima1 = loppu[0] + ' ' + loppu[1]
    leima2 = loppu[2].split('.')[0]
    featuresetlist = []
    if 'S1' in filename:
        featuresetlist.append('S1')
    if 'S2' in filename:
        featuresetlist.append('S2')
    if not featuresetlist: # if filename does not reveal anything, then in our case it is only 'S2'
        featuresetlist.append('S2')

    if ('S1' in featuresetlist) and ('S2' in featuresetlist):
        leima3 = 'S1S2'
    if ('S1' in featuresetlist) and ('S2' not in featuresetlist):
        leima3 = 'S1'
    if ('S1' not in featuresetlist) and ('S2' in featuresetlist):
        leima3 = 'S2'

    if '123450' in filename:  
        classes = [   
         'Plough',
         'Conservation tillage',
         'Winter crop',
         'Grass',
         'Stubble',
         'Stubble + companion crop'
        ]
        classnr = '123450'
    elif '1234' in filename:
        classes = [
         'Conservation tillage',
         'Winter crop',
         'Grass',
         'Stubble']
        classnr = '1234'
    elif '1230' in filename:  
        classes = [   
         'Plough',
         'Conservation tillage + stubble',
         'Winter crop',
         'Grass'
        ]
        classnr = '1230'
    elif '123' in filename:
        classes = [        
         'Winter crop',
         'Grass',
         'Conservation tillage + stubble']
        classnr = '123'
    else:
        
        print('Which classes do we have here? Cannot make up my mind...')
        
        
    # reading the file:
    with open(file, 'rb') as f:
        preds = pickle.load(f)
    
    accuracy = balanced_accuracy_score(preds[0], preds[1]) # balanced_accuracy_score(y_true, y_pred)

    if accuracy > 0.62:
        print(accuracy)
        print(f'Preds: {np.unique(preds[1])}') # predictions
        print(f'True: {np.unique(preds[0])}') # true
        filename = os.path.join(outputpath, 'ConfusionMatrix-' + classifiername + '-' + classnr + '-' + timeMethod + '-' + leima3 + '-' + str(accuracy) + '.eps')
        print(f'Saving into {filename}')
        print(f'Originates from {file}')
        # order: ytest, test_predictions
        plot_confusion_matrix(preds[0], preds[1], 'Predicted label', 'In situ label', classes, normalize=True, filename = filename)
        
    else:
        continue



