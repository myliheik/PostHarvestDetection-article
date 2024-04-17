"""
2023-02-14 MY classification on in situ data. do random train-test split.
2023-12-08 MY: test must not include S2 duplicates, because they have duplicated S1 data. 
        So in case of 'S1-S2-2D' and 'S1-S2-3D', sweep all duplicates into training set.
        
2023-12-11 SA & VI
2023-12-15 feature wise SA ei eroja, joten kokeillaan band set SA
2024-01-22 incl. InSAR features 

Copy of : 26-sensitivityAnalysis-timeseries-group-article-InSAR.py
2024-02-24 Classes 12340

number of classes:

12340: plough vs. conservation tillage vs. winter crop vs. grass vs. stubble (41)

plough: 0
conservation tillage: 1
winter crop: 2
grass: 3
stubble (41): 4


TIME SERIES, TCN:

python ../python/26-experiment4.py \
-o /Users/myliheik/Documents/myVEGCOVER/vegcover2023/sensitivityAnalysis/experiment4 \
-i /Users/myliheik/Documents/myVEGCOVER/vegcover2023/cloudless/dataStack-insitu/ard-S1-S2-3D_2020-2021-2022-2023.npz \
-n /Users/myliheik/Documents/myVEGCOVER/vegcover2023/InSAR/20230408/ard2023InSAR.pkl \
-c 12340

python ../python/26-experiment4.py \
-o /Users/myliheik/Documents/myVEGCOVER/vegcover2023/sensitivityAnalysis/experiment4 \
-i /Users/myliheik/Documents/myVEGCOVER/vegcover2023/sensitivityAnalysis/experiment4/ard-S1-S2-3D_2020-2021-2022-2023.npz \
-n /Users/myliheik/Documents/myVEGCOVER/vegcover2023/sensitivityAnalysis/experiment4/ard2023InSAR.pkl \
-c 12340


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

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense, GlobalMaxPooling1D, Dropout, Input
from tensorflow.keras import Model


from tcn import TCN

#### EDIT PARAMETERS:

normmethod = 'l1'

nbfilters = 65
kernelsize = 5 # maybe more is more, revisit time close to 3 days?
dilatiot = [1, 2, 4, 8, 13]


doClassWeights = True
epokit = 200
batchit = 64
    
callback = EarlyStopping(monitor='loss', patience = 5)

t = time.localtime()
#timeString  = time.strftime("%Y-%m-%d", t)

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
    print(f'Shape of S1S2 dataset: {df0.shape}\n')

    ######################################## Get meta info:    

    if 'ard' in inputfile:
        metafile = inputfile.replace('ard-S1-S2-2D', 'parcelID-ard').replace('ard-S1-S2-3D', 'parcelID-ard').replace('ard-S1-3D', 'parcelID-ard').replace('ard-S2-3D', 'parcelID-ard').replace('npz', 'pkl')

    else:
        metafile = inputfile.replace('normalized3D', 'parcelID').replace('npz', 'pkl')

    # test:
    metattest = pd.DataFrame(utils._load_intensities(metafile), columns = ['parcelID'])
    metatMeta = metattest.merge(omameta, how = 'left', on = 'parcelID')
        
    return df0, metatMeta

######################################################################################################################

def munchS1S2toInSAR(array0, metatMeta, dfInSAR):

    masktest = metatMeta['target'].isna()
    if metatMeta['target'].isna().any():
        print(f'There are {metatMeta["target"].isna().sum()} target information missing in test set!')
    if len(metatMeta) != len(array0):
        print(f'Meta file and test data not matching: {len(metatMeta)}, {len(array0)}')
        
        
    # make a mask for metatMeta['parcelID'] and dfInSAR['parcelID']
    # take only InSAR parcels:
    InSARmask = metatMeta['parcelID'].isin(dfInSAR['parcelID'])    
    metatMeta2 = metatMeta.loc[InSARmask, ['parcelID', 'target']]
    array1 = array0[InSARmask, :, :]

    parcels = pd.DataFrame(metatMeta2[['parcelID', 'target']]) 
    dfInSAR2 = parcels.merge(dfInSAR, how = 'left') # take parcels from metatMeta, now in the same order
    
    print(f"Test if S1S2 parcels and InSAR parcels are in the same order: {(dfInSAR2['parcelID'].values == metatMeta2['parcelID'].values).all()}")

    # Merge array1 and dfInSAR2
    # dfInSAR2 into time series (zero-padding)

    vv = dfInSAR2.loc[:, dfInSAR2.columns.str.contains('VV')].values
    vh = dfInSAR2.loc[:, dfInSAR2.columns.str.contains('VH')].values
    InSARarray = np.dstack((vv,vh))
    InSARarray2 = np.swapaxes(InSARarray, 1, 2)
    InSARarray3 = np.expand_dims(InSARarray2, axis=1)
    # back to 3D:
    m,n = InSARarray3.shape[:2]
    InSARarray4 = InSARarray3.reshape(m,n,-1) 
    
    
    # How many time points in S1S2?:
    timepoints = array1.shape[1]
    print('Zero-pad InSAR data (only 1 timepoint) to have the same number of time points as S2 data.')
    array4padded = np.hstack((InSARarray4, np.zeros((m, timepoints-1, InSARarray4.shape[2]))))
    print(f'Shape of array4padded: {array4padded.shape} \n Shape of S1S2 array: {array1.shape}')
    # Stack InSAR & S1S2 in 3D:
    InSARdata3D = np.dstack((array1, array4padded))

    print(f'InSAR shape in 3D: (obs, bands, features), reshaped, with time dimension: {InSARarray4.shape}, zero-padded: {array4padded.shape}, stacked with S1S2 array: {InSARdata3D.shape}\n\n')

    # duplicates due to overlapping orbits in S2 -> duplicates into training set!
    # mask duplicates, need to be placed into training set later:
    maskDuplicated = metatMeta2['parcelID'].duplicated(keep = False) # False : Mark all duplicates as True.
    print(f"NA in target: {metatMeta2['target'].isna().sum()}")
    print(f"Not NA in target and duplicated: {(~metatMeta2['target'].isna() & maskDuplicated).sum()}")
    print(f'There are {maskDuplicated.sum()} duplicated parcels. These will be placed into training set.') 


    metatMeta3 = metatMeta2.assign(split = 'test')
    metatMeta3['split'][maskDuplicated] = 'train'

    maskSplit = np.in1d(metatMeta3['split'], 'train', invert = False)

    if len(InSARdata3D.shape) == 3: # 3D
        setti2 = '3D'
        xtrain = InSARdata3D[maskSplit,:,:]  
        ytrain = metatMeta3['target'][maskSplit]
    else:    
        print(f'Something is wrong, data is not in 3D, but in {len(InSARdata3D.shape)}')
        
        
    ######################################################################################################################
    # test set -> take the reverse of duplicated: 

    maskSplit = np.in1d(metatMeta3['split'], 'train', invert = True)

    xtest = InSARdata3D[maskSplit,:,:]  
    ytest = metatMeta3['target'][maskSplit]
        
                
    return xtrain, xtest, ytrain, ytest

def chooseClasses(xtrain, xtest, ytrain, ytest, classes):        
    ######################### pick classes and readjust:

    if classes == ['12340']: # 0 vs. 1 vs. 2 vs. 3 vs. 41
        ########## TRAIN SET:  
        xtrain0 = xtrain.copy()
        ytrain0 = ytrain
        # Choose only classes 0,1,2,3,41. Not 42 and 5:
        maskA = np.in1d(ytrain0, [42, 5], invert = True)
        xtrain = xtrain0[maskA,:,:]   
        ytrain00 = ytrain0[maskA]

        maskB = np.in1d(ytrain00, [41], invert = False)
        ytrain00[maskB] = 4

        ytrain = ytrain00 

        if len(xtrain) != len(ytrain):
            print(f'Length of xtrain does not match ytrain: {len(xtrain)} is not {len(ytrain)}')

    elif classes == ['123450']:
        ytrain0 = ytrain

        maskB = np.in1d(ytrain0, [41, 42], invert = False)
        ytrain0[maskB] = 4     
        ytrain = ytrain0

    else:
        # all classes
        print(f'Do all classes.')

    if len(xtrain) != len(ytrain):
        print(f'Length of xtrain does not match ytrain: {len(xtrain)} is not {len(ytrain)}')
        
        

        
    ######################### test set: pick classes and readjust:


    if classes == ['12340']: # 0 vs. 1 vs. 2 vs. 3 vs. 41
        ########## TRAIN SET:  
        xtest0 = xtest.copy()
        ytest0 = ytest.values
        # Choose only classes 0,1,2,3,41. Not 42 and 5:
        maskA = np.in1d(ytest0, [42, 5], invert = True)
        xtest = xtest0[maskA,:,:]   
        ytest00 = ytest0[maskA]

        maskB = np.in1d(ytest00, [41], invert = False)
        ytest00[maskB] = 4              

        ytest = ytest00

        if len(xtest) != len(ytest):
            print(f'Length of xtrain does not match ytrain: {len(xtest)} is not {len(ytest)}')                


    elif classes == ['123450']:
        ytest0 = ytest.values

        maskB = np.in1d(ytest0, [41, 42], invert = False)
        ytest0[maskB] = 4     
        ytest = ytest0


    else:
        # all classes
        print(f'Do all classes.')

    if len(xtest) != len(ytest):
        print(f'Length of xtest does not match ytest: {len(xtest)} is not {len(ytest)}')



    return xtrain, xtest, ytrain, ytest


    
    
    


######################################################################################################################


def classify(xtrain, xtest, ytrain, ytest, setti, timeString, tunniste, outputpath, results, classes, dataset1, dataset2, dataset3, inputfile):
    # Classifiers:

    print(np.unique(ytrain))
    
    ################################
    # Balancing:
    classweights = dict(zip(np.unique(ytrain), class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(ytrain), 
                y = ytrain))) 
        
    ################################            
        
    for i in range(1): # repeated 
    
        if setti == 'timeseries':
            resultsTCN = []
            name = 'TCN'  

            ################################
            ############## 3D CLASSIFICATION:
            ################################

            ################################
            ############## Leave-1-out sensitivity analysis
            ################################

            # slicing out 32 piirrettä (bins) xtrain ja xtest

            #excludedFeatures = ['SAR', 'VIS', 'NIR', 'SWIR']
            #excludedFeatures = ['VIS', 'NIR', 'SWIR', 'COH', 'SARCOH']
            #excludedFeatures = ['SAR']
            #excludedFeatures = ['SAR', 'VIS', 'NIR', 'SWIR', 'COH', 'SARCOH', 'All']
            excludedFeatures = ['All'] # all included, actually


            for groupp in excludedFeatures:

                if groupp == 'SAR':
                    #### Leave out SAR:
                    xtrainSA = xtrain[:, :, 64:]
                    xtestSA = xtest[:, :, 64:]

                elif groupp == 'VIS': # 2,3,4
                    VISmask = np.repeat(True,xtrain.shape[2])
                    VISmask[64:160] = False
                    xtrainSA = xtrain[:, :, VISmask]
                    xtestSA = xtest[:, :, VISmask]

                elif groupp == 'NIR': # 5,6,7,8,8A
                    NIRmask = np.repeat(True,xtrain.shape[2])
                    NIRmask[160:288] = False
                    NIRmask[352:384] = False                
                    xtrainSA = xtrain[:, :, NIRmask]
                    xtestSA = xtest[:, :, NIRmask]

                elif groupp == 'SWIR': # 11,12
                    SWIRmask = np.repeat(True,xtrain.shape[2])
                    SWIRmask[288:352] = False
                    xtrainSA = xtrain[:, :, SWIRmask]
                    xtestSA = xtest[:, :, SWIRmask]

                elif groupp == 'COH': # COHVV, COHVH
                    COHmask = np.repeat(True,xtrain.shape[2])
                    COHmask[384:] = False
                    xtrainSA = xtrain[:, :, COHmask]
                    xtestSA = xtest[:, :, COHmask]

                elif groupp == 'SARCOH': # VV, VH, COHVV, COHVH
                    SARCOHmask = np.repeat(True,xtrain.shape[2])
                    SARCOHmask[384:] = False
                    SARCOHmask[:64] = False
                    xtrainSA = xtrain[:, :, SARCOHmask]
                    xtestSA = xtest[:, :, SARCOHmask]
                    
                else:
                    xtrainSA = xtrain
                    xtestSA = xtest

                nClasses = len(np.unique(ytrain))
                print(f'Number of classes: {nClasses}')

                tcn_layer = TCN(input_shape=(None, xtrainSA.shape[2]), nb_filters = nbfilters, padding = 'causal', kernel_size = kernelsize, 
                            nb_stacks=1, dilations = dilatiot, 
                            return_sequences=True
                           )

                classifier = Sequential([
                tcn_layer,
                GlobalMaxPooling1D(),
                Dense(nClasses, activation='softmax')
                ])

                #classifier.summary()
                classifier.compile(optimizer = Adam(learning_rate=0.001, amsgrad=True, epsilon = 1e-7),
                           loss = 'sparse_categorical_crossentropy',
                           metrics =['accuracy'])
                print(f'Train {name}...')
                
                fpmodelfig = os.path.join(outputpath, "TCN-model-in-experiment4-300dpi.png")
                print(f'Save TCN model graph in {fpmodelfig}')
                plot_model(classifier, fpmodelfig, dpi = 300, show_shapes=True)
                kaput
                if doClassWeights:
                    normmethod = 'l1-balanced'
                    classifier.fit(xtrainSA, ytrain, validation_split=0.2, epochs=epokit, batch_size = batchit, verbose = 0, class_weight=classweights, callbacks=[callback])        
                else:
                    normmethod = 'l1'
                    classifier.fit(xtrainSA, ytrain, validation_split=0.2, epochs=epokit, batch_size = batchit, verbose = 0, callbacks=[callback])        

                y_test_predicted = classifier.predict(xtestSA)
                test_predictions = np.argmax(y_test_predicted, axis = 1)
                #print(test_predictions)
                accu = accuracy_score(ytest, test_predictions)
                bala = balanced_accuracy_score(ytest, test_predictions)
                reca = recall_score(ytest, test_predictions, average='weighted')
                prec = precision_score(ytest, test_predictions, average='weighted')
                resultsTCN.append([name, dataset1, dataset2, dataset3, setti, groupp, normmethod, accu, prec, reca, bala, classes, len(ytrain), len(ytest), inputfile])       
                print(resultsTCN)
                # save predictions:
                outfile = os.path.join(outputpath, tunniste + '-predictions-' + groupp + '-' + name + '-' + normmethod + '-' + setti + '-' +  classes[0] + '-' + timeString + '.pkl')
                utils.save_intensities(outfile, np.array([ytest, test_predictions]))



            # save
            dfTCN = pd.DataFrame(resultsTCN, columns=["Classifier", "S1", "S2", "S1S2", "Time", "Excluded feature", "Normalized", "OA", "Precision", "Recall", "Balanced acc.", "Classes", "Nr. of training", "Nr. of testing", "Dataset"])
            outfile = os.path.join(outputpath, tunniste + '-results-' + timeString + '.csv')
            print('\nSaving results to file: ' + outfile)


            dfTCN.to_csv(outfile, header = True, index = False)


def main(args):

    try:
        if not args.inputfile or not args.outputpath:
            raise Exception('Missing input file or output directory argument. Try --help .')

        print(f'\n26-experiment4.py')
        print(f'\nSensitivity analysis on TCN classifiers.')
   
        out_dir_path = Path(os.path.expanduser(args.outputpath))
        out_dir_path.mkdir(parents=True, exist_ok=True)
               
        results = []
        featuresetlist = []

        head, tail = os.path.split(args.inputfile)
        tunniste = tail.split('.')[0]

        setti = 'timeseries'      
        timeString  = time.strftime("%Y-%m-%d-%H:%M-TCN-timeseries-leave1out", t)

        if doClassWeights:
            normmethod = 'l1-balanced'
          
        else: 
            normmethod = 'l1'
          
        # Read S1&S2:
        df0, metatMeta = initialization(args.inputfile)
        # Read InSAR:
        dfInSAR, insarfeatures = load_InSAR(args.insarfile)       
        # Merge S1S2 and InSAR data:
        xtrain, xtest, ytrain, ytest = munchS1S2toInSAR(df0, metatMeta, dfInSAR) # when  S1 included needs to sweep duplicates into training set
        # Pick classes of interest:
        xtrain, xtest, ytrain, ytest = chooseClasses(xtrain, xtest, ytrain, ytest, args.classes) 


        dataset1 = 0; dataset2 = 0; dataset3 = 1 # S1S2 included
        print(f'Shape of the train set: {xtrain.shape} \n')
        print(f'Case {args.classes}:\nLength of the data set: {len(df0)} \n')
        print(f'Class distribution in data set:\n {np.unique(metatMeta["target"], return_counts=True)}')
        print(f'Class distribution in train set:\n {np.unique(ytrain, return_counts=True)}')
        print(f'Class distribution in test set:\n {np.unique(ytest, return_counts=True)}')
        
        ######################################################################################################################
        # For time series data only:
        classify(xtrain, xtest, ytrain, ytest, setti, timeString, tunniste, out_dir_path, results, args.classes[0], dataset1, dataset2, dataset3, args.inputfile)
        
            
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
    parser.add_argument('-n', '--insarfile',
                        type = str,
                        help = 'ARD file for InSAR data.')

    parser.add_argument('-o', '--outputpath',
                        type = str,
                        help = 'Directory for output file.',
                        default = '.')
    parser.add_argument('-c', '--classes', action='store', dest='classes',
                       type=str, nargs='*', default=['1240'],
                       help="Classes to use, e.g. -c 1240 123450")
  
    
    args = parser.parse_args()
    main(args)    

