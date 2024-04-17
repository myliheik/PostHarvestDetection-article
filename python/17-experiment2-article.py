"""
2024-02-23 MY
Article Experiment 2: 


RUN:
# 1. S2 3D for TCN, duplicates in training set:
python ../python/17-experiment2-article.py -i /Users/myliheik/Documents/myVEGCOVER/vegcover2023/results/experiment2/xtrain-ard-S2-3D_2020-2021-2022-2023-12340-timeseries.npz \
-o /Users/myliheik/Documents/myVEGCOVER/vegcover2023/results/experiment2/20240223

# 2. MLP S2 fusion, duplicates in training set:
python ../python/17-experiment2-article.py -i /Users/myliheik/Documents/myVEGCOVER/vegcover2023/results/experiment2/xtrain-normalized3D_2020-2021-2022-2023-12340-Fusion.npz \
-o /Users/myliheik/Documents/myVEGCOVER/vegcover2023/results/experiment2/20240223

# 3. S2 3D no duplicates for TCN:
python ../python/17-experiment2-article.py -i /Users/myliheik/Documents/myVEGCOVER/vegcover2023/results/experiment2/xtrain-ard-S2-3D-noDuplicates_2020-2021-2022-2023-12340-timeseries.npz \
-o /Users/myliheik/Documents/myVEGCOVER/vegcover2023/results/experiment2/20240223

# 4. S1S2 3D no duplicates for TCN:
python ../python/17-experiment2-article.py -i /Users/myliheik/Documents/myVEGCOVER/vegcover2023/results/experiment2/xtrain-ard-S1-S2-3D-noDuplicates_2020-2021-2022-2023-12340-timeseries.npz \
-o /Users/myliheik/Documents/myVEGCOVER/vegcover2023/results/experiment2/20240223

# 5. MLP S1S2 fusion, duplicates in training set:
python ../python/17-experiment2-article.py -i /Users/myliheik/Documents/myVEGCOVER/vegcover2023/results/experiment2/xtrain-ard-S1-S2-2D_2020-2021-2022-2023-12340-Fusion.npz \
-o /Users/myliheik/Documents/myVEGCOVER/vegcover2023/results/experiment2/20240223

"""

import os.path
from pathlib import Path
import time
import argparse
import textwrap
import pandas as pd
import numpy as np
import utils
import datetime

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalMaxPooling1D, Dropout, Input, Concatenate
from tensorflow.keras import Model

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report, recall_score, balanced_accuracy_score
from sklearn.utils import class_weight

from tcn import TCN

# EDIT hyperparameters:

nbfilters = 65
kernelsize = 5 # maybe more is more, revisit time close to 3 days?
dilatiot = [1, 2, 4, 8, 13]


doClassWeights = True
epokit = 200
batchit = 64
    
callback = EarlyStopping(monitor='loss', patience = 5)

t = time.localtime()

# MLP:
iterations = 200
alphaMLP = 0.001

# TCN:
alphaTCN = 0.001





######################################################################################################################
# run TCN:
def classify(xtrain, xtest, ytrain, ytest, setti, timeString, tunniste, outputpath, results, classes, dataset1, dataset2, dataset3, inputfile):
    print('3D classification')        
    nClasses = len(np.unique(ytrain))
    print(f'Number of classes: {nClasses}')
    
    ################################
    # Balancing:
    classweights = dict(zip(np.unique(ytrain), class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(ytrain), 
                y = ytrain))) 
        
    ################################   
    name = 'TCN'   
    name = f'TCN-{nbfilters}-filters-{kernelsize}-kernelsize-{epokit}-epocs'

    tcn_layer = TCN(input_shape=(None, xtrain.shape[2]), nb_filters = nbfilters, padding = 'causal', kernel_size = kernelsize, 
                nb_stacks=1, dilations = dilatiot, 
                return_sequences=True
               )

    classifier = Sequential([
                    tcn_layer,
                    GlobalMaxPooling1D(),
                    Dense(nClasses, activation='softmax')
                    ])

    #classifier.summary()
    classifier.compile(optimizer = Adam(learning_rate = 0.001, amsgrad = True, epsilon = 1e-7),
               loss = 'sparse_categorical_crossentropy',
               metrics =['accuracy'])
    print(f'Train {name}...')

    if doClassWeights:
        normmethod = 'l1-balanced'
        history = classifier.fit(xtrain, ytrain, validation_split = 0.2, epochs = epokit, batch_size = batchit, verbose = 0, class_weight = classweights, callbacks = [callback])        
    else:
        normmethod = 'l1'
        history = classifier.fit(xtrain, ytrain, validation_split = 0.2, epochs = epokit, batch_size = batchit, verbose = 0, callbacks = [callback])        

    y_test_predicted = classifier.predict(xtest)
    test_predictions = np.argmax(y_test_predicted, axis = 1)
    #print(test_predictions)
    accu = accuracy_score(ytest, test_predictions)
    bala = balanced_accuracy_score(ytest, test_predictions)
    reca = recall_score(ytest, test_predictions, average='weighted')
    prec = precision_score(ytest, test_predictions, average='weighted')
    results.append([name, dataset1, dataset2, dataset3, setti, normmethod, accu, prec, reca, bala, classes, len(ytrain), len(ytest), inputfile,
                   round(history.history['accuracy'][-1], 2), round(history.history['val_accuracy'][-1], 2)])       
    print(results)
    # save predictions:
    outfile = os.path.join(outputpath, tunniste + '-predictions-' + name + '-' + normmethod + '-' + setti + '-' +  classes + '-' + timeString + '.pkl')
    utils.save_intensities(outfile, np.array([ytest, test_predictions]))

    return results


######################################################################################################################

def vanillaMLP(xtrain, xtest, ytrain, ytest, outputpath, tunniste, normmethod, setti, classes, timeString):   

    hidden_layersize = (1024, 920, 740, 512, 256, 128,)
    hidden_layers = 6
    
    name = f'MLP-{hidden_layers}-hidden_layers-{iterations}-iterations-{alphaMLP}-alpha'

    clf = MLPClassifier(max_iter = iterations, epsilon = 1e-7, verbose = False, alpha = alphaMLP, 
                        hidden_layer_sizes = hidden_layersize, early_stopping = True)

    clf.fit(xtrain, ytrain)  

    y_pred = clf.predict(xtest)
    accu = clf.score(xtest, ytest)
    bala = balanced_accuracy_score(ytest, y_pred)
    reca = recall_score(ytest, y_pred, average='weighted')
    prec = precision_score(ytest, y_pred, average='weighted')

    # save predictions:
    outfile = os.path.join(outputpath, tunniste + '-predictions-' + name + '-' + normmethod + '-' + setti + '-' +  classes + '-' + timeString + '.pkl')
    utils.save_intensities(outfile, np.array([ytest, y_pred]))
    
    
    
    return name, accu, prec, reca, bala, None, None

######################################################################################################################



######################################################################################################################
def main(args):

    try:
        if not args.inputfile or not args.outputpath:
            raise Exception('Missing input file or output directory argument. Try --help .')

        print(f'\n17-experiment2-article.py')
        print(f'\nRun several classifiers.')
   
        out_dir_path = Path(os.path.expanduser(args.outputpath))
        out_dir_path.mkdir(parents=True, exist_ok=True)
               
        featuresetlist = []
        results = []
            
        head, tail = os.path.split(args.inputfile)
        tunniste = tail.split('.')[0]
        

        if 'timeseries' in args.inputfile:
            setti = 'timeseries'  
        if 'Fusion' in args.inputfile:
            setti = 'Fusion'

        if doClassWeights:
            normmethod = 'l1-balanced'
          
        else: 
            normmethod = 'l1'
          
        # Read in ARD data:
        xtrain = utils.load_npintensities(args.inputfile)
        
        fpxtest = os.path.join(head, tail.replace('xtrain', 'xtest'))
        xtest = utils.load_npintensities(fpxtest)
        
        fpytrain = os.path.join(head, tail.replace('xtrain', 'ytrain'))
        ytrain = utils.load_npintensities(fpytrain)
        
        fpytest = os.path.join(head, tail.replace('xtrain', 'ytest'))
        ytest = utils.load_npintensities(fpytest)
 
        

        if 'S1' in args.inputfile:
            featuresetlist.append('S1')
        if 'S2' in args.inputfile:
            featuresetlist.append('S2')
        if not featuresetlist: # if filename does not reveal anything, then in our case it is only 'S2'
            featuresetlist.append('S2')

        if ('S1' in featuresetlist) and ('S2' in featuresetlist):
            dataset1 = 0; dataset2 = 0; dataset3 = 1
        if ('S1' in featuresetlist) and ('S2' not in featuresetlist):
            dataset1 = 1; dataset2 = 0; dataset3 = 0
        if ('S1' not in featuresetlist) and ('S2' in featuresetlist):
            dataset1 = 0; dataset2 = 1; dataset3 = 0    
            

        classes = tail.split('-')[-2]

        print(f'\nShape of the train set: {xtrain.shape} \n')
        print(f'Class distribution in train set:\n {np.unique(ytrain, return_counts=True)}')
        print(f'Class distribution in test set:\n {np.unique(ytest, return_counts=True)}')
        
        ######################################################################################################################
        results = []
        
        if 'normalized3D' in args.inputfile:
            timeString  = time.strftime("%Y-%m-%d-%H:%M-MLP-fusion", t)
            dataset1 = 0; dataset2 = 1; dataset3 = 0  
            if setti == 'Fusion':
                # make 2D from S2 3D:
                xtrain = xtrain.reshape(xtrain.shape[0], -1)
                xtest = xtest.reshape(xtest.shape[0], -1)
                print(f'Converted S2 data from 3D to 2D:\n {xtrain.shape} and {xtest.shape}')
            
            # Vanilla MLP            
            name, accu, prec, reca, bala, trainacc, valacc = vanillaMLP(xtrain, xtest, ytrain, ytest, out_dir_path, tunniste, normmethod, setti, classes, timeString) 
            results.append([name, dataset1, dataset2, dataset3, setti, normmethod, accu, prec, reca, bala, classes, len(ytrain), len(ytest), args.inputfile, None, None])
         
        elif 'Fusion' in args.inputfile:
            timeString  = time.strftime("%Y-%m-%d-%H:%M-MLP-fusion", t)
            # Vanilla MLP            
            name, accu, prec, reca, bala, trainacc, valacc = vanillaMLP(xtrain, xtest, ytrain, ytest, out_dir_path, tunniste, normmethod, setti, classes, timeString)
            results.append([name, dataset1, dataset2, dataset3, setti, normmethod, accu, prec, reca, bala, classes, len(ytrain), len(ytest), args.inputfile, None, None])            
            
        elif 'timeseries' in args.inputfile:   
            timeString  = time.strftime("%Y-%m-%d-%H:%M-TCN-timeseries-insituonly", t)
            results = classify(xtrain, xtest, ytrain, ytest, setti, timeString, tunniste, out_dir_path, results, classes, dataset1, dataset2, dataset3, args.inputfile)  
            
            
        df = pd.DataFrame(results, columns=['Classifier', 'S1', 'S2', 'S1S2', 'Time', "Normalized", "OA", "Precision", "Recall", "Balanced acc.", "Classes", "Nr. of training", "Nr. of testing", "Dataset", "Training accuracy", "Validation accuracy"])
        print(df)
        outfile = os.path.join(out_dir_path, tunniste + '-results-' + timeString + '.csv')
        print('\nSaving results to file: ' + outfile)
        df.to_csv(outfile, header = True, index = False)               

            
        print('Done.')

    except Exception as e:
        print('\n\nUnable to read input or write out results. Check prerequisites and see exception output below.')
        parser.print_help()
        raise e
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=textwrap.dedent(__doc__))

    parser.add_argument('-i', '--inputfile',
                        type = str,
                        help = 'Train X file for input.')
    parser.add_argument('-o', '--outputpath',
                        type = str,
                        help = 'Directory for output file.',
                        default = '.')
    
    args = parser.parse_args()
    main(args)    


