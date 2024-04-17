"""
2023-10-09 RF and feature importance
2024-01-02 InSAR
2D: obs, bands (vv*32 + vh*32 + B02*32 + B03*32 + ... + B12*32 + B8A*32) 384 features

+ InSAR features


DEPENDENCIES:


RUN:

Fusion (in 2D):

python 26-RF-experiment4.py \
-o /Users/myliheik/Documents/myVEGCOVER/vegcover2023/experiment4/featureImportance \
-i /Users/myliheik/Documents/myVEGCOVER/vegcover2023/cloudless/fused-insitu/ard-S1-S2-2D_2020-2021-2022-2023.npz \
-j /Users/myliheik/Documents/myVEGCOVER/vegcover2023/InSAR/20230408/ard2023InSAR.pkl \
-c 12340


# Article:
1240: 0 vs. 1 vs. green vs. 41

# IFS:
1230: 1) Winter crop, 2) Grass and Stubble with companion crop, 3) Stubble and Conservation tillage, 4) Ploughing



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


# Classifiers:

from sklearn.ensemble import RandomForestClassifier


#### EDIT PARAMETERS:
classifiernames = [
"RandomForest"
]

# parametrit koeteltu 23.4.2021
classifiers = [
RandomForestClassifier(max_depth=None, n_estimators=500, max_features='sqrt', n_jobs = -1, class_weight = 'balanced', oob_score = True, verbose = False)
] 
 
normmethod = 'l1'

doClassWeights = True

t = time.localtime()
#timeString  = time.strftime("%Y-%m-%d", t)
timeString  = time.strftime("%Y-%m-%d-%H:%M-RF-2D-Fusion-S1S2-InSAR", t)
    
def load_InSAR(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    insarfeatures = data.columns.str.replace('bin.*_', '', regex = True)[1:]
    return data, insarfeatures
    

def initialization(inputfile):

    # in situ metat:
    omameta = pd.read_csv('/Users/myliheik/Documents/myVEGCOVER/vegcover2023/metafiles/omameta.csv').drop_duplicates(subset=['parcelID'])
    # bigRefe meta löytyy täältä: 
    megameta = pd.read_csv('/Users/myliheik/Documents/myVEGCOVER/vegcover2023/metafiles/meta1234VarmatFilteredOutInSitu.csv')    

    df0 = utils.load_npintensities(inputfile)

    ######################################## Check dimensions:
    print(f'Shape of dataset: {df0.shape}\n')

    ######################################## Get meta info:    

    if 'ard' in inputfile:
        metafile = inputfile.replace('ard-S1-S2-2D', 'parcelID-ard').replace('ard-S1-S2-3D', 'parcelID-ard').replace('ard-S1-3D', 'parcelID-ard').replace('ard-S2-3D', 'parcelID-ard').replace('npz', 'pkl')

    else:
        metafile = inputfile.replace('normalized3D', 'parcelID').replace('npz', 'pkl')

    # test:
    metattest = pd.DataFrame(utils._load_intensities(metafile), columns = ['parcelID'])
    metatMeta = metattest.merge(omameta, how = 'left', on = 'parcelID')
        
    return df0, metatMeta


def munchS1S2toInSAR(df0, metatMeta, dfInSAR):

    masktest = metatMeta['target'].isna()
    if metatMeta['target'].isna().any():
        print(f'There are {metatMeta["target"].isna().sum()} target information missing in test set!')
    if len(metatMeta) != len(df0):
        print(f'Meta file and test data not matching: {len(metatMeta)}, {len(df0)}')
        
        
    # merge meta + array to a DataFrame:
    df00 = pd.DataFrame(df0)
    df01 = pd.concat([metatMeta[['parcelID', 'target']], df00], axis = 1)
    df02 = df01.merge(dfInSAR, on = 'parcelID', how = 'left') # merge to InSAR
    df03 = df02.dropna() # drop InSAR NAs
    df033 = df03.reset_index()
    
    # duplicates due to overlapping orbits in S2 -> duplicates into training set!
    # mask duplicates, need to be placed into training set later:
    maskDuplicated = df033['parcelID'].duplicated(keep = False) # False : Mark all duplicates as True.
    print(f"NA in target: {df033['target'].isna().sum()}")
    print(f"Not NA in target and duplicated: {(~df033['target'].isna() & maskDuplicated).sum()}")
    print(f'There are {maskDuplicated.sum()} duplicated parcels. These will be placed into training set.') 
    

    df04 = df033.assign(split = 'test')
    df04['split'][maskDuplicated] = 'train'

    # train:
    xtrain0 = df04.loc[df04['split'] == 'train',:]  
    xtrain = xtrain0.drop(['index', 'parcelID', 'target', 'split'], axis = 1).values
    ytrain = df04['target'][df04['split'] == 'train']

    # test:
    xtest0 = df04.loc[df04['split'] == 'test',:]  
    xtest = xtest0.drop(['index', 'parcelID', 'target', 'split'], axis = 1).values
    ytest = df04['target'][df04['split'] == 'test']


    return xtrain, ytrain, xtest, ytest



def munchClasses(xtrain, ytrain, xtest, ytest, classes):
    ######################### pick classes and readjust:
    # Works for Fusion:
    # train:
    xdata = xtrain
    ydata = ytrain
    if classes == ['1234']: # 1 vs. 2 vs. 3 vs. 4
        ########## TRAIN SET:  
        xdata0 = xdata.copy()
        ydata0 = ydata.values
        # Choose only classes 1,2,3 and 4:
        maskA = np.in1d(ydata0, [0,42,5], invert = True)
        xdata = xdata0[maskA] # numpy masking
        ydata00 = ydata0[maskA]

        maskB = np.in1d(ydata00, [41], invert = False)
        ydata00[maskB] = 4
        ydata = ydata00 - 1 # ei vaikuta, vaikka oliskin vain 1,2,3,4, nyt 0,1,2,3,4

        if len(xdata) != len(ydata):
            print(f'Length of xdata does not match ydata: {len(xdata)} is not {len(ydata)}')

    elif classes == ['12340']: # 1 vs. 2 vs. 3 vs. 4 vs. 0
        ########## TRAIN SET:  
        xdata0 = xdata.copy()
        ydata0 = ydata.values
        # Choose only classes 0,1,2,3,41. Not 42 and 5:
        maskA = np.in1d(ydata0, [42, 5], invert = True)
        xdata = xdata0[maskA]   
        ydata00 = ydata0[maskA]

        maskB = np.in1d(ydata00, [41], invert = False)
        ydata00[maskB] = 4

        ydata = ydata00 


    elif classes == ['123']: #  2 vs. 3 vs. 1 & 4
        ########## TRAIN SET:  
        ydata0 = ydata.values
        maskB = np.in1d(ydata0, [2, 3], invert = True)
        ydata0[maskB] = 1     
        ydata = ydata0 - 1


    elif classes == ['123450']:
        ydata0 = ydata.values
        maskB = np.in1d(ydata0, [41, 42], invert = False)
        ydata0[maskB] = 4     
        ydata = ydata0

    elif classes == ['1230']:
        xdata0 = xdata.copy()
        ydata0 = ydata.values
        maskB = np.in1d(ydata0, [41, 42, 4, 5], invert = False)
        ydata0[maskB] = 1   
        ydata = ydata0
        
    elif classes == ['1240']: # 0 vs. 1 vs. green vs. 41
        ########## TRAIN SET:  
        xdata0 = xdata.copy()
        ydata0 = ydata.values
        # Choose only classes 0,1,2,3,41. Not 42 and 5:
        maskA = np.in1d(ydata0, [42, 5], invert = True)
        xdata = xdata0[maskA]   
        ydata00 = ydata0[maskA]

        maskB = np.in1d(ydata00, [41], invert = False)
        ydata00[maskB] = 4

        maskGG = np.in1d(ydata00, [2, 3], invert = False) # join classes 2,3
        ydata00[maskGG] = 2

        maskFF = np.in1d(ydata00, [4], invert = False) # class weights wants labels in order
        ydata00[maskFF] = 3                

        ydata = ydata00 

        
    else:
        # all classes
        print(f'Do all classes.')

    if len(xdata) != len(ydata):
        print(f'Length of xdata does not match ydata: {len(xdata)} is not {len(ydata)}')

    xtrain2 = xdata
    ytrain2 = ydata
    
    ########################## test:
    xdata = xtest
    ydata = ytest
    if classes == ['1234']: # 1 vs. 2 vs. 3 vs. 4
        ########## TRAIN SET:  
        xdata0 = xdata.copy()
        ydata0 = ydata.values
        # Choose only classes 1,2,3 and 4:
        maskA = np.in1d(ydata0, [0,42,5], invert = True)
        xdata = xdata0[maskA] # numpy masking
        ydata00 = ydata0[maskA]

        maskB = np.in1d(ydata00, [41], invert = False)
        ydata00[maskB] = 4
        ydata = ydata00 - 1 # ei vaikuta, vaikka oliskin vain 1,2,3,4, nyt 0,1,2,3

        if len(xdata) != len(ydata):
            print(f'Length of xdata does not match ydata: {len(xdata)} is not {len(ydata)}')

    elif classes == ['12340']: # 1 vs. 2 vs. 3 vs. 4 vs. 0
        ########## TRAIN SET:  
        xdata0 = xdata.copy()
        ydata0 = ydata.values
        # Choose only classes 0,1,2,3,41. Not 42 and 5:
        maskA = np.in1d(ydata0, [42, 5], invert = True)
        xdata = xdata0[maskA]   
        ydata00 = ydata0[maskA]

        maskB = np.in1d(ydata00, [41], invert = False)
        ydata00[maskB] = 4

        ydata = ydata00 


    elif classes == ['123']: #  2 vs. 3 vs. 1 & 4
        ########## TRAIN SET:  
        ydata0 = ydata.values
        maskB = np.in1d(ydata0, [2, 3], invert = True)
        ydata0[maskB] = 1     
        ydata = ydata0 - 1


    elif classes == ['123450']:
        ydata0 = ydata.values
        maskB = np.in1d(ydata0, [41, 42], invert = False)
        ydata0[maskB] = 4     
        ydata = ydata0

    elif classes == ['1230']:
        xdata0 = xdata.copy()
        ydata0 = ydata.values
        maskB = np.in1d(ydata0, [41, 42, 4, 5], invert = False)
        ydata0[maskB] = 1   
        ydata = ydata0
        
    elif classes == ['1240']: # 0 vs. 1 vs. green vs. 41
        ########## TRAIN SET:  
        xdata0 = xdata.copy()
        ydata0 = ydata.values
        # Choose only classes 0,1,2,3,41. Not 42 and 5:
        maskA = np.in1d(ydata0, [42, 5], invert = True)
        xdata = xdata0[maskA]   
        ydata00 = ydata0[maskA]

        maskB = np.in1d(ydata00, [41], invert = False)
        ydata00[maskB] = 4

        maskGG = np.in1d(ydata00, [2, 3], invert = False) # join classes 2,3
        ydata00[maskGG] = 2

        maskFF = np.in1d(ydata00, [4], invert = False) # class weights wants labels in order
        ydata00[maskFF] = 3                

        ydata = ydata00 
        
    else:
        # all classes
        print(f'Do all classes.')

    if len(xdata) != len(ydata):
        print(f'Length of xdata does not match ydata: {len(xdata)} is not {len(ydata)}')

    xtest2 = xdata
    ytest2 = ydata
    
    return xtrain2, ytrain2, xtest2, ytest2

def runRF(xtrain, ytrain, xtest, ytest, outputpath, classes, insarfeatures, tunniste, inputfile):
        
    dataset1 = 0; dataset2 = 0; dataset3 = 1             
    print(f'Data in train set: {len(ytrain)}')
    print(f'Data in test set: {len(ytest)}')    
    print(f'Class distribution in train set:\n {np.unique(ytrain, return_counts=True)}')
    print(f'Class distribution in test set:\n {np.unique(ytest, return_counts=True)}')
    
    ################################
    # Balancing:
    classweights = dict(zip(np.unique(ytrain), class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(ytrain), 
                y = ytrain))) 
        
    ################################
    ############## CLASSIFICATION:
    ################################
    
    setti = '2D'
    normmethod = 'l1'
    error_rate = []
    results = []
    features = ['VV', 'VH', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12', 'B8A']
    feature_names0 = np.repeat(features, 32)
    feature_names = feature_names0.tolist() + insarfeatures.values.tolist()

    # iterate or not:            
    for i in range(19):
    #for i in range(1):
        print(f'Train RF. \nIterate round {i} ...')
        # iterate over classifiers
        for name, clf in zip(classifiernames, classifiers):
            #print(name, dataset1, dataset2, dataset3, setti, classes)
            #if doClassWeights: # ei toimi näille, pitää tehdä manuaalisesti
            #    normmethod = 'l1-balanced'
            #    clf.fit(xtrain, ytrain, class_weight=classweights)
            #else:
            normmethod = 'l1'

            clf.fit(xtrain, ytrain)  
            

            importances = clf.feature_importances_
            

            std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
          
            forest_importances = pd.Series(importances, index = feature_names)
            print(forest_importances)
            outfileimportance = os.path.join(outputpath, tunniste + '-importance-' + name + '-' + normmethod + '-' + setti + '-' +  classes[0] + '.csv')
            
            forest_importances.to_csv(outfileimportance, header = True, index = False)



            if (name == 'BalancedRF') | (name == 'RandomForest'):
                # Record the OOB error for each `n_estimators=i` setting.
                oob_error = 1 - clf.oob_score_
                error_rate.append(oob_error)
                #outfile = os.path.join(outputpath, tunniste + '-oob-' + name + '-' + normmethod + '-' + setti + '-' +  classes[0] + '-' + timeString + '.pkl')
                #utils.save_intensities(outfile, error_rate)
                print(error_rate)      

            y_pred = clf.predict(xtest)
            accu = clf.score(xtest, ytest)
            bala = balanced_accuracy_score(ytest, y_pred)
            reca = recall_score(ytest, y_pred, average='weighted')
            prec = precision_score(ytest, y_pred, average='weighted')
            #accu = None; prec = None; reca = None; bala = None
            results.append([name, dataset1, dataset2, dataset3, setti, normmethod, accu, prec, reca, bala, classes[0], len(ytrain), len(ytest), inputfile])
            print(name, dataset1, dataset2, dataset3, setti, normmethod, accu, prec, reca, bala, classes[0], len(ytrain), len(ytest))
            # save predictions:
            outfile = os.path.join(outputpath, tunniste + '-predictions-' + name + '-' + normmethod + '-' + setti + '-' +  classes[0] + '-' + timeString + '.pkl')
            utils.save_intensities(outfile, np.array([ytest, y_pred]))




        print(results)            
        df = pd.DataFrame(results, columns=['Classifier', 'S1', 'S2', 'S1S2', 'Time', "Normalized", "OA", "Precision", "Recall", "Balanced acc.", "Classes", "Nr. of training", "Nr. of testing", "Data set"])
        outfile = os.path.join(outputpath, tunniste + '-results-' + timeString + '.csv')

        print('\nSaving results to file: ' + outfile)
        #print(df[[name, dataset1, dataset2, dataset3, setti, excludedFeature, normmethod, accu, prec, reca, bala]])
        df.to_csv(outfile, header = True, index = False)
                     
# MAIN:


def main(args):

    try:
        if not args.inputfile or not args.outputpath:
            raise Exception('Missing input file or output directory argument. Try --help .')

        print(f'\n26-RF-experiment4.py')
        print(f'\nRun RF classifiers for variable importance.')
   
        out_dir_path = Path(os.path.expanduser(args.outputpath))
        out_dir_path.mkdir(parents=True, exist_ok=True)
        
        fpS1S2 = args.inputfile
        tunniste = fpS1S2.split('/')[-1].replace('.npz', '-InSAR2023')
       
        # Read InSAR:
        dfInSAR, insarfeatures = load_InSAR(args.insarfile)
        # Read S1S2:
        df0, metatMeta = initialization(fpS1S2)
        # Merge S1S2 to InSAR:
        xtrain0, ytrain0, xtest0, ytest0 = munchS1S2toInSAR(df0, metatMeta, dfInSAR)
        # Munch classes:
        xtrain, ytrain, xtest, ytest = munchClasses(xtrain0, ytrain0, xtest0, ytest0, args.classes)
        
        
        runRF(xtrain, ytrain, xtest, ytest, out_dir_path, args.classes, insarfeatures, tunniste, fpS1S2)        
        
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
                        help = 'S1S2 ARD file for input.')
    parser.add_argument('-j', '--insarfile',
                        type = str,
                        help = 'InSAR ARD file for input.')

    parser.add_argument('-o', '--outputpath',
                        type = str,
                        help = 'Directory for output file.',
                        default = '.')
    parser.add_argument('-c', '--classes', action='store', dest='classes',
                       type=str, nargs='*', default=['1234'],
                       help="Classes to use, e.g. -c 1234 10 0")
        
    args = parser.parse_args()
    main(args)    

