"""
2024-02-17 MY
Article Experiment 3: 


RUN:
# 

python 17-experiment3-article.py -i /Users/myliheik/Documents/myVEGCOVER/vegcover2023/results/experiment3/xtrain-ard-S1-S2-3D_2020-2021-2022-2023-12340-timeseries.npz \ 
-o /Users/myliheik/Documents/myVEGCOVER/vegcover2023/results/experiment3/20240217




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
#hidden_layersize = 128
iterations = 200
alphaMLP = 0.01

# TCN:
alphaTCN = 0.001

# TCNhiddenMLP:
alphahidden = 0.001

doTCN = True
doTCNensemble = True
doTCNhiddenMLP = False

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

    print(classifier.summary())
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

# TCN to ensemble models:
def classifyTCN(xtrain0, xtest0, ytrain, ytest):

    xtrain = xtrain0[:,:,64:]
    xtest = xtest0[:,:,64:]

    print(np.unique(ytrain))
    
    ################################
    # Balancing:
    classweights = dict(zip(np.unique(ytrain), class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(ytrain), 
                y = ytrain))) 

    print('TCNensemble classification')        
    nClasses = len(np.unique(ytrain))
    print(f'Number of classes: {nClasses}')

    name = 'TCNensemble'   
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
    classifier.compile(optimizer = Adam(learning_rate = alphaTCN, amsgrad = True, epsilon = 1e-7),
               loss = 'sparse_categorical_crossentropy',
               metrics =['accuracy'])
    print(f'Train {name}...')

    if doClassWeights:
        normmethod = 'l1-balanced'
        classifier.fit(xtrain, ytrain, validation_split = 0.2, epochs = epokit, batch_size = batchit, verbose = 0, class_weight = classweights, callbacks=[callback])        
    else:
        normmethod = 'l1'
        classifier.fit(xtrain, ytrain, validation_split = 0.2, epochs = epokit, batch_size = batchit, verbose = 0, callbacks = [callback])        
    
    return classifier



######################################################################################################################


def MLP(classifier, xtrain, xtest, ytrain, ytest):   
    
    #hidden_layersize = (1024, 920, 740, 512, 256, 256, 256, 256, 256, 256, 256, 256, 128, 128, 128, 128,) # 16
    #hidden_layersize = (512, 256, 256, 128, 128,) # 5
    #hidden_layersize = (512, 256, 128, 100,) # 4
    hidden_layersize = (256, 128, 100,) # 3
    hidden_layers = 3
    
    name = f'TCNtoMLP-{hidden_layers}-hidden_layers-{iterations}-iterations-{alphaMLP}-alpha'

    xtrain2D = xtrain[:,0,:64]
    xtest2D = xtest[:,0,:64]

    xtrain3D = xtrain[:,:,64:]
    xtest3D = xtest[:,:,64:]

    tcn_predicted_train = classifier.predict(xtrain3D)
    tcn_predicted_test = classifier.predict(xtest3D)

    # yhdistetään:
    xtrainMLP = np.hstack((xtrain2D, tcn_predicted_train))
    xtestMLP = np.hstack((xtest2D, tcn_predicted_test))

    clf = MLPClassifier(max_iter = iterations, epsilon = 1e-7, verbose = False, alpha = alphaMLP, 
                        hidden_layer_sizes = hidden_layersize, early_stopping = True)

    clf.fit(xtrainMLP, ytrain)  

    y_pred = clf.predict(xtestMLP)
    accu = clf.score(xtestMLP, ytest)
    bala = balanced_accuracy_score(ytest, y_pred)
    reca = recall_score(ytest, y_pred, average='weighted')
    prec = precision_score(ytest, y_pred, average='weighted')
    print(name, accu, prec, reca, bala)

    return name, accu, prec, reca, bala, None, None


######################################################################################################################


def classifierTCNhiddenMLP(xtrain, xtest, ytrain, ytest):
    
    dense_layers = 3
    layer_size = 128
    dropout = 0.2
    
    # yhdistetään time series TCN ja MLP layer

    name = f'TCNhiddenMLP-{dense_layers - 1}-hidden_layers-{layer_size}-neurons-{dropout}-dropoutRate'

    nClasses = len(np.unique(ytrain))
    
    ################################
    # Balancing:
    classweights = dict(zip(np.unique(ytrain), class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(ytrain), 
                y = ytrain))) 

    
    xtrain2D = xtrain[:,0,:64]
    xtest2D = xtest[:,0,:64]

    xtrain3D = xtrain[:,:,64:]
    xtest3D = xtest[:,:,64:]
    
    I = Input(shape=(None, xtrain3D.shape[2]))

    tcn_layer = TCN(nb_filters = nbfilters, padding = 'causal', kernel_size = kernelsize, 
                nb_stacks = 1, dilations = dilatiot, 
                return_sequences=True)(I)

    hidden2 = GlobalMaxPooling1D()(tcn_layer)
    tcn_output = Dense(nClasses, activation='softmax')(hidden2)

    mlp_input = Input(shape=(xtrain2D.shape[1])) # MLP in

    x = Concatenate()([tcn_output, mlp_input])

    for l in range(dense_layers - 1):
        x = Dense(layer_size, activation = "relu")(x)

        x = Dropout(dropout)(x)
    
    mlp_out = Dense(nClasses, activation='softmax')(x) # MLP out

    classifier = Model(inputs = [I, mlp_input], outputs = [mlp_out])
    classifier.summary()
    classifier.compile(optimizer = Adam(learning_rate = alphahidden, amsgrad = True, epsilon = 1e-7),
               loss = 'sparse_categorical_crossentropy',
               metrics =['accuracy'])

    print(f'Train {name}...')

    #classifier.summary()

    if doClassWeights:
        normmethod = 'l1-balanced'
        history = classifier.fit([xtrain3D, xtrain2D], ytrain, validation_split = 0.2, epochs = epokit, batch_size = batchit, verbose = 0, class_weight=classweights, callbacks=[callback])        
    else:
        normmethod = 'l1'
        history = classifier.fit([xtrain3D, xtrain2D], ytrain, validation_split = 0.2, epochs = epokit, batch_size = batchit, verbose = 0, callbacks = [callback]) 
        
    y_test_predicted = classifier.predict([xtest3D, xtest2D])
    test_predictions = np.argmax(y_test_predicted, axis = 1)
    #print(test_predictions)
    accu = accuracy_score(ytest, test_predictions)
    bala = balanced_accuracy_score(ytest, test_predictions)
    reca = recall_score(ytest, test_predictions, average = 'weighted')
    prec = precision_score(ytest, test_predictions, average = 'weighted')
    #results.append([name, dataset1, dataset2, dataset3, setti, normmethod, accu, prec, reca, bala, classes, nbfilters, kernelsize, dilatiot])         
    print(name, accu, prec, reca, bala, round(history.history['accuracy'][-1], 2), round(history.history['val_accuracy'][-1], 2))
          
    return name, accu, prec, reca, bala, round(history.history['accuracy'][-1], 2), round(history.history['val_accuracy'][-1], 2)


######################################################################################################################
def main(args):

    try:
        if not args.inputfile or not args.outputpath:
            raise Exception('Missing input file or output directory argument. Try --help .')

        print(f'\n17-experiment3-article.py')
        print(f'\nRun several classifiers.')
   
        out_dir_path = Path(os.path.expanduser(args.outputpath))
        out_dir_path.mkdir(parents=True, exist_ok=True)
               
        featuresetlist = []
        results = []
            
        head, tail = os.path.split(args.inputfile)
        tunniste = tail.split('.')[0]
        
        # must be time series:            
        setti = 'timeseries'  

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

        print(f'\nLength of the train set: {len(xtrain)} \n')
        print(f'Class distribution in train set:\n {np.unique(ytrain, return_counts=True)}')
        print(f'Class distribution in test set:\n {np.unique(ytest, return_counts=True)}')

        ######################################################################################################################
        if doTCN:
            timeString  = time.strftime("%Y-%m-%d-%H:%M-TCN-timeseries-insituonly", t)

            results0 = classify(xtrain, xtest, ytrain, ytest, setti, timeString, tunniste, out_dir_path, results, classes, dataset1, dataset2, dataset3, args.inputfile)
           
            df = pd.DataFrame(results0, columns=['Classifier', 'S1', 'S2', 'S1S2', 'Time', "Normalized", "OA", "Precision", "Recall", "Balanced acc.", "Classes", "Nr. of training", "Nr. of testing", "Dataset", "Training accuracy", "Validation accuracy"])
            print(df) 
            outfile = os.path.join(out_dir_path, tunniste + '-results-' + timeString + '.csv')
            print('\nSaving results to file: ' + outfile)
            df.to_csv(outfile, header = True, index = False)   
        else:
            print(f'Skipping vanilla TCN this time...')

        if setti == 'timeseries' and doTCNensemble and 'S1' in args.inputfile:
            results = []
            timeString  = time.strftime("%Y-%m-%d-%H:%M-TCNtoMLP-timeseries-insituonly", t)

            # we train TCN for 3D S2 data and then classics (MLP) for 2D S1:
            classifier = classifyTCN(xtrain, xtest, ytrain, ytest)
            
            name, accu, prec, reca, bala, trainacc, valacc = MLP(classifier, xtrain, xtest, ytrain, ytest)
            results.append([name, dataset1, dataset2, dataset3, setti, normmethod, accu, prec, reca, bala, classes, len(ytrain), len(ytest), args.inputfile, trainacc, valacc])          
            df = pd.DataFrame(results, columns=['Classifier', 'S1', 'S2', 'S1S2', 'Time', "Normalized", "OA", "Precision", "Recall", "Balanced acc.", "Classes", "Nr. of training", "Nr. of testing", "Dataset", "Training accuracy", "Validation accuracy"])
            outfile = os.path.join(out_dir_path, tunniste + '-results-' + timeString + '.csv')
            print('\nSaving results to file: ' + outfile)
            #print(df[[name, dataset1, dataset2, dataset3, setti, normmethod, accu, prec, reca, bala]])
            df.to_csv(outfile, header = True, index = False)
            print(df)
        else:
            print(f'Skipping TCNtoMLP this time...')
            
        # we train TCN layers for 3D S2 data followed by MLP layers for 2D S1:
        if setti == 'timeseries' and doTCNhiddenMLP and 'S1' in args.inputfile:
            results = []
            timeString  = time.strftime("%Y-%m-%d-%H:%M-TCNhiddenMLP-timeseries-insituonly", t)

            name, accu, prec, reca, bala, trainacc, valacc = classifierTCNhiddenMLP(xtrain, xtest, ytrain, ytest)
            results.append([name, dataset1, dataset2, dataset3, setti, normmethod, accu, prec, reca, bala, classes, len(ytrain), len(ytest), args.inputfile, trainacc, valacc])
            df = pd.DataFrame(results, columns=['Classifier', 'S1', 'S2', 'S1S2', 'Time', "Normalized", "OA", "Precision", "Recall", "Balanced acc.", "Classes", "Nr. of training", "Nr. of testing", "Dataset", "Training accuracy", "Validation accuracy"])
            outfile = os.path.join(out_dir_path, tunniste + '-results-' + timeString + '.csv')
            print('\nSaving results to file: ' + outfile)
            df.to_csv(outfile, header = True, index = False)
            print(df)
        else:
            print(f'Skipping TCNhiddenMLP this time...')
            
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


