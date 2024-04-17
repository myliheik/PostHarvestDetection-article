"""
2023-02-14 MY classification on in situ data. do random train-test split.
2024-02-25

MOSAIC & Fusion


Number of classes:

123450: 6 classes (plough, conservative tillage, autumn crop, grass, stubble, stubble+companion crop)
12340: 5 classes (plough, conservative tillage, autumn crop, grass, stubble + stubble+companion crop)
1230: 4 classes (plough, conservation tillage + stubble + stubble+companion crop, autumn crop, grass)



0: plough vs. other classes
10: 3 classes (plough, conservative tillage, others)
20: 2 classes (plough vs. autumn crop)
230: autumn crop & grass vs. plough
340: plough vs. grass vs. stubble
1234: conservative tillage, autumn crop, grass, stubble
123: autumn crop, grass, others



RUN:

python ../python/17-experiment1.py \
-o /Users/myliheik/Documents/myVEGCOVER/vegcover2023/results/experiment1/mosaicS1S2 \
-i /Users/myliheik/Documents/myVEGCOVER/vegcover2023/results/experiment1/mosaicS1S2/xtrain-ard-S1-S2ind-2D-123450-Mosaics.npz \
-c 123450

OR:

python ../python/17-experiment1.py \
-o /Users/myliheik/Documents/myVEGCOVER/vegcover2023/results/experiment1/fusedS1S2 \
-i /Users/myliheik/Documents/myVEGCOVER/vegcover2023/results/experiment1/fusedS1S2/xtrain-ard-S1-S2-2D-noDuplicates_2020-2021-2022-2023-123450-Fusion.npz \
-c 123450


"""
# nr of cores to use:
#njobs=16

import os.path
from pathlib import Path
import time
import argparse
import textwrap
import pandas as pd
import numpy as np
import pickle
import utils

from math import sqrt, floor
from sklearn.metrics import accuracy_score, precision_score, classification_report, recall_score, balanced_accuracy_score, confusion_matrix
from sklearn.utils import class_weight

from sklearn.model_selection import train_test_split

# Classifiers:
from sklearn.neural_network import MLPClassifier


#### EDIT PARAMETERS:

normmethod = 'l1'
doClassWeights = False

# MLP:
iterations = 200
alphaMLP = 0.001
#alphaMLP = 0.01

t = time.localtime()
#timeString  = time.strftime("%Y-%m-%d", t)
    

def runClassifiers(xtrain, ytrain, xtest, ytest, outputpath, classes, setti, dataset1, dataset2, dataset3, tunniste):

    results = []

    timeString  = time.strftime("%Y-%m-%d-%H:%M-MLP-2D", t)

    
    ################################
    # Balancing: # not applicable to MLPClassifier
    #classweights = dict(zip(np.unique(ytrain), class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(ytrain), 
    #            y = ytrain))) 
    
    ################################            

    # iterate or not:            
    #for i in range(10):
    for i in range(2):
        if setti == 'Fusion':
            #hidden_layersize = (256, 128, 100,) # 3
            #hidden_layers = 3
            #hidden_layersize = (1024, 920, 740, 512, 256, 128,)
            #hidden_layers = 6
            hidden_layersize = (1024, 128,) # best
            hidden_layers = 2
            #hidden_layersize = (2024, 1024, 36,) 
            #hidden_layers = 3
        elif setti == 'Mosaic':
            #hidden_layersize = (1000,) 
            #hidden_layers = 1
            hidden_layersize = (1024, 36,) # best
            hidden_layers = 2
            #hidden_layersize = (1024, 120, 36) 
            #hidden_layers = 3
            
        else:
            print(f'Dataset is not either Fusion or Mosaic. What is this {setti}?')
                
        name = f'MLP-{hidden_layers}-hidden_layers-{iterations}-iterations-{alphaMLP}-alpha'

        clf = MLPClassifier(max_iter = iterations, epsilon = 1e-7, verbose = False, alpha = alphaMLP, 
                            hidden_layer_sizes = hidden_layersize, early_stopping = True)

        clf.fit(xtrain, ytrain)  

        y_pred = clf.predict(xtest)
        accu = clf.score(xtest, ytest)
        bala = balanced_accuracy_score(ytest, y_pred)
        reca = recall_score(ytest, y_pred, average='weighted')
        prec = precision_score(ytest, y_pred, average='weighted')

        print(name, dataset1, dataset2, dataset3, setti, normmethod, accu, prec, reca, bala, classes, len(ytrain), len(ytest), args.inputfile)
        # save predictions:
        outfile = os.path.join(outputpath, tunniste + '-predictions-' + name + '-' + normmethod + '-' + setti + '-' +  classes[0] + '-' + timeString + '.pkl')
        utils.save_intensities(outfile, np.array([ytest, y_pred]))

        results.append([name, dataset1, dataset2, dataset3, setti, normmethod, accu, prec, reca, bala, classes[0], len(ytrain), len(ytest), args.inputfile])            


      
    print(results)    

    df = pd.DataFrame(results, columns=['Classifier', 'S1', 'S2', 'S1S2', 'Time', "Normalized", "OA", "Precision", "Recall", "Balanced acc.", "Classes", "Nr. of training", "Nr. of testing", "Dataset"])
    outfile = os.path.join(outputpath, tunniste + '-results-' + timeString + '.csv')
    print('\nSaving results to file: ' + outfile)
    #print(df[[name, dataset1, dataset2, dataset3, setti, normmethod, accu, prec, reca, bala]])
    df.to_csv(outfile, header = True, index = False)
                     
# MAIN:


def main(args):

    try:
        if not args.inputfile or not args.outputpath:
            raise Exception('Missing input file or output directory argument. Try --help .')

        print(f'\n17-experiment1.py')
        print(f'\nRun MLP classifiers.')
   
        out_dir_path = Path(os.path.expanduser(args.outputpath))
        out_dir_path.mkdir(parents=True, exist_ok=True)
        
        featuresetlist = []
            
        head, tail = os.path.split(args.inputfile)
        tunniste = tail.split('.')[0]
        
        if 'Mosaic' in args.inputfile:
            setti = 'Mosaic'  
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
            

        #classes = tail.split('-')[-2]
        #print(classes)
        print(f'Shape of train: {xtrain.shape}')
        print(f'Shape of test: {xtest.shape}')
        
        print(f'\nLength of the train set: {len(xtrain)} \n')
        print(f'Class distribution in train set:\n {np.unique(ytrain, return_counts=True)}')
        print(f'Class distribution in test set:\n {np.unique(ytest, return_counts=True)}')
   
        runClassifiers(xtrain, ytrain, xtest, ytest, out_dir_path, args.classes, setti, dataset1, dataset2, dataset3, tunniste)        
        
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
                        help = 'ARD file for input.')
    parser.add_argument('-o', '--outputpath',
                        type = str,
                        help = 'Directory for output file.',
                        default = '.')
    parser.add_argument('-c', '--classes', action='store', dest='classes',
                       type=str, nargs='*', default=['123450'],
                       help="Classes to use, e.g. -c 1234 10 0")

        
    args = parser.parse_args()
    main(args)    

