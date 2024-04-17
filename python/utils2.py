"""
2024-02-17 MY 

Utility functions for preprocessing data for VEGCOVER ML-modelling.

"""

import numpy as np
import os.path
import re
import pandas as pd
from sklearn.model_selection import train_test_split
import utils

######################################################################################################################


def munchS1S2(df0, metatMeta, classes):

    masktest = metatMeta['target'].isna()
    if metatMeta['target'].isna().any():
        print(f'There are {metatMeta["target"].isna().sum()} target information missing in test set!')
    if len(metatMeta) != len(df0):
        print(f'Meta file and test data not matching: {len(metatMeta)}, {len(df0)}')

    # duplicates due to overlapping orbits in S2 -> duplicates into training set!
    # mask duplicates, neeed to be placed into training set later:
    maskDuplicated = metatMeta['parcelID'].duplicated(keep = False) # False : Mark all duplicates as True.
    print(f"NA in target: {metatMeta['target'].isna().sum()}")
    print(f"Not NA in target and duplicated: {(~metatMeta['target'].isna() & maskDuplicated).sum()}")
    print(f'There are {maskDuplicated.sum()} duplicated parcels. These will be placed into training set.') 
    
    metatMeta2 = metatMeta.assign(split = 'test')
    metatMeta2['split'][maskDuplicated] = 'train'


    maskSplit = np.in1d(metatMeta2['split'], 'train', invert = False)

    # remove also NAs:
    if len(df0.shape) == 3: # 3D
        setti2 = '3D'
        xtrain = df0[maskSplit,:,:]  
        ytrain = metatMeta2['target'][maskSplit]

    else:
        setti2 = '2D'
        # train:
        xtrain = df0[maskSplit,:]  
        ytrain = metatMeta2['target'][maskSplit]


    ######################### pick classes and readjust:
    # Works for Fusion:
    if setti2 == '2D':

        if classes == ['1240']: # 0 vs. 1 vs. green vs. 41
            ########## TRAIN SET:  
            xtrain0 = xtrain.copy()
            ytrain0 = ytrain.values
            # Choose only classes 0,1,2,3,41. Not 42 and 5:
            maskA = np.in1d(ytrain0, [42, 5], invert = True)
            xtrain = xtrain0[maskA] # numpy masking
            ytrain00 = ytrain0[maskA]

            maskB = np.in1d(ytrain00, [41], invert = False)
            ytrain00[maskB] = 4

            maskGG = np.in1d(ytrain00, [2, 3], invert = False) # join classes 2,3
            ytrain00[maskGG] = 2

            maskFF = np.in1d(ytrain00, [4], invert = False) # class weights wants labels in order
            ytrain00[maskFF] = 3                

            ytrain = ytrain00

            if len(xtrain) != len(ytrain):
                print(f'Length of xtrain does not match ytrain: {len(xtrain)} is not {len(ytrain)}')
                
        elif classes == ['12340']: # 0 vs. 1 vs. 2 vs. 3 vs. 41
            ########## TRAIN SET:  
            xtrain0 = xtrain.copy()
            ytrain0 = ytrain.values
            # Choose only classes 0,1,2,3,41. Not 42 and 5:
            maskA = np.in1d(ytrain0, [42, 5], invert = True)
            xtrain = xtrain0[maskA] # numpy masking
            ytrain00 = ytrain0[maskA]

            maskB = np.in1d(ytrain00, [41], invert = False)
            ytrain00[maskB] = 4

            ytrain = ytrain00

            if len(xtrain) != len(ytrain):
                print(f'Length of xtrain does not match ytrain: {len(xtrain)} is not {len(ytrain)}')
                
        elif classes == ['123450']:
            ytrain0 = ytrain.values
            maskB = np.in1d(ytrain0, [41, 42], invert = False)
            ytrain0[maskB] = 4     
            ytrain = ytrain0

        else:
            # all classes
            print(f'Do all classes.')

        if len(xtrain) != len(ytrain):
            print(f'Length of xtrain does not match ytrain: {len(xtrain)} is not {len(ytrain)}')

    else: # timeseries

        if classes == ['1240']: # 0 vs. 1 vs. green vs. 41
            ########## TRAIN SET:  
            xtrain0 = xtrain.copy()
            ytrain0 = ytrain.values
            # Choose only classes 0,1,2,3,41. Not 42 and 5:
            maskA = np.in1d(ytrain0, [42, 5], invert = True)
            xtrain = xtrain0[maskA,:,:]   
            ytrain00 = ytrain0[maskA]

            maskB = np.in1d(ytrain00, [41], invert = False)
            ytrain00[maskB] = 4

            maskGG = np.in1d(ytrain00, [2, 3], invert = False) # join classes 2,3
            ytrain00[maskGG] = 2

            maskFF = np.in1d(ytrain00, [4], invert = False) # class weights wants labels in order
            ytrain00[maskFF] = 3                

            ytrain = ytrain00 

            if len(xtrain) != len(ytrain):
                print(f'Length of xtrain does not match ytrain: {len(xtrain)} is not {len(ytrain)}')

        elif classes == ['12340']: # 0 vs. 1 vs. 2 vs. 3 vs. 41
            ########## TRAIN SET:  
            xtrain0 = xtrain.copy()
            ytrain0 = ytrain.values
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
            ytrain0 = ytrain.values

            maskB = np.in1d(ytrain0, [41, 42], invert = False)
            ytrain0[maskB] = 4     
            ytrain = ytrain0

        else:
            # all classes
            print(f'Do all classes.')

        if len(xtrain) != len(ytrain):
            print(f'Length of xtrain does not match ytrain: {len(xtrain)} is not {len(ytrain)}')
    ######################################################################################################################
    # test set -> take the reverse of duplicated: 

    maskSplit = np.in1d(metatMeta2['split'], 'train', invert = True)

    # remove also NAs:
    if len(df0.shape) == 3: # 3D
        setti2 = '3D'
        xtest = df0[maskSplit,:,:]  
        ytest = metatMeta2['target'][maskSplit]

    else:
        setti2 = '2D'
        # train:
        xtest = df0[maskSplit,:]  
        ytest = metatMeta2['target'][maskSplit]

    ######################### pick classes and readjust:
    # Works for Fusion:
    if setti2 == '2D':

        if classes == ['1240']: # 0 vs. 1 vs. green vs. 41
            ########## TRAIN SET:  
            xtest0 = xtest.copy()
            ytest0 = ytest.values
            # Choose only classes 0,1,2,3,41. Not 42 and 5:
            maskA = np.in1d(ytest0, [42, 5], invert = True)
            xtest = xtest0[maskA]   
            ytest00 = ytest0[maskA]

            maskB = np.in1d(ytest00, [41], invert = False)
            ytest00[maskB] = 4

            maskGG = np.in1d(ytest00, [2, 3], invert = False) # join classes 2,3
            ytest00[maskGG] = 2

            maskFF = np.in1d(ytest00, [4], invert = False) # class weights wants labels in order
            ytest00[maskFF] = 3                

            ytest = ytest00 

            if len(xtest) != len(ytest):
                print(f'Length of xtrain does not match ytrain: {len(xtest)} is not {len(ytest)}')  
                
        elif classes == ['12340']: # 0 vs. 1 vs. 2 vs. 3 vs. 41
            ########## TRAIN SET:  
            xtest0 = xtest.copy()
            ytest0 = ytest.values
            # Choose only classes 0,1,2,3,41. Not 42 and 5:
            maskA = np.in1d(ytest0, [42, 5], invert = True)
            xtest = xtest0[maskA]   
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

    else: # timeseries

        if classes == ['1240']: # 0 vs. 1 vs. green vs. 41
            ########## TRAIN SET:  
            xtest0 = xtest.copy()
            ytest0 = ytest.values
            # Choose only classes 0,1,2,3,41. Not 42 and 5:
            maskA = np.in1d(ytest0, [42, 5], invert = True)
            xtest = xtest0[maskA,:,:]   
            ytest00 = ytest0[maskA]

            maskB = np.in1d(ytest00, [41], invert = False)
            ytest00[maskB] = 4

            maskGG = np.in1d(ytest00, [2, 3], invert = False) # join classes 2,3
            ytest00[maskGG] = 2

            maskFF = np.in1d(ytest00, [4], invert = False) # class weights wants labels in order
            ytest00[maskFF] = 3                

            ytest = ytest00

            if len(xtest) != len(ytest):
                print(f'Length of xtrain does not match ytrain: {len(xtest)} is not {len(ytest)}')                

        elif classes == ['12340']: # 0 vs. 1 vs. 2 vs. 3 vs. 41
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


######################################################################################################################

def munchS2(df0, metatMeta, classes):# when just S2, no need to drop duplicates OR if S1S2 AND duplicates already removed
    
    masktest = metatMeta['target'].isna()
    if metatMeta['target'].isna().any():
        print(f'There are {metatMeta["target"].isna().sum()} target information missing!')
    if len(metatMeta) != len(df0):
        print(f'Meta file and dataset not matching: {len(metatMeta)}, {len(df0)}')
    print(metatMeta['target'].isna().any())

    # remove NAs:
    if len(df0.shape) == 3: # 3D
        setti2 = '3D'
        xdata = df0[np.logical_not(masktest),:,:]   
        ydata = metatMeta['target'][np.logical_not(masktest)]

    else:
        setti2 = '2D'
        # train:
        xdata = df0[np.logical_not(masktest),:]   
        ydata = metatMeta['target'][np.logical_not(masktest)]



    # Works for Fusion:
    if setti2 == '2D':

        if classes == ['1240']: # 0 vs. 1 vs. green vs. 41
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

            if len(xdata) != len(ydata):
                print(f'Length of xtrain does not match ytrain: {len(xdata)} is not {len(ydata)}')      
                
        elif classes == ['12340']: # 0 vs. 1 vs. 2 vs. 3 vs. 41
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

            if len(xdata) != len(ydata):
                print(f'Length of xtrain does not match ytrain: {len(xdata)} is not {len(ydata)}')                

        elif classes == ['123450']:
            ydata0 = ydata.values
            maskB = np.in1d(ydata0, [41, 42], invert = False)
            ydata0[maskB] = 4     
            ydata = ydata0

        else:
            # all classes
            print(f'Do all classes.')

        if len(xdata) != len(ydata):
            print(f'Length of xdata does not match ydata: {len(xdata)} is not {len(ydata)}')

    else: # timeseries

        if classes == ['1240']: # 0 vs. 1 vs. green vs. 41
            ########## TRAIN SET:  
            xdata0 = xdata.copy()
            ydata0 = ydata.values
            # Choose only classes 0,1,2,3,41. Not 42 and 5:
            maskA = np.in1d(ydata0, [42, 5], invert = True)
            xdata = xdata0[maskA,:,:]   
            ydata00 = ydata0[maskA]

            maskB = np.in1d(ydata00, [41], invert = False)
            ydata00[maskB] = 4

            maskGG = np.in1d(ydata00, [2, 3], invert = False) # join classes 2,3
            ydata00[maskGG] = 2

            maskFF = np.in1d(ydata00, [4], invert = False) # class weights wants labels in order
            ydata00[maskFF] = 3                

            ydata = ydata00 

            if len(xdata) != len(ydata):
                print(f'Length of xtrain does not match ytrain: {len(xdata)} is not {len(ydata)}')     
                
        elif classes == ['12340']: # 0 vs. 1 vs. green vs. 41
            ########## TRAIN SET:  
            xdata0 = xdata.copy()
            ydata0 = ydata.values
            # Choose only classes 0,1,2,3,41. Not 42 and 5:
            maskA = np.in1d(ydata0, [42, 5], invert = True)
            xdata = xdata0[maskA,:,:]   
            ydata00 = ydata0[maskA]

            maskB = np.in1d(ydata00, [41], invert = False)
            ydata00[maskB] = 4

            ydata = ydata00 

            if len(xdata) != len(ydata):
                print(f'Length of xtrain does not match ytrain: {len(xdata)} is not {len(ydata)}')                

        elif classes == ['123450']:
            ydata0 = ydata.values

            maskB = np.in1d(ydata0, [41, 42], invert = False)
            ydata0[maskB] = 4     
            ydata = ydata0

        else:
            # all classes
            print(f'Do all classes.')

        if len(xdata) != len(ydata):
            print(f'Length of xdata does not match ydata: {len(xdata)} is not {len(ydata)}')



    ##################################
    # random split:    
    xtrain, xtest, ytrain, ytest = train_test_split(xdata, ydata, test_size = 0.25, stratify = ydata)
    return xtrain, xtest, ytrain, ytest

def munchMosaics(inputfile, classes):
    metafile = inputfile.replace('ard-S1-S2ind-2D', 'parcelID-ard')
    ydata = utils.load_npintensities(metafile)

    if classes == ['123450']: 
        ydata0 = ydata.copy()
        maskB = np.in1d(ydata0, [41, 42], invert = False)
        ydata0[maskB] = 4     
        ydata = ydata0
    else:
        print('Experiment 1 is only for all classes (123450)!')

    xdata = utils.load_npintensities(inputfile)
    # random split:    
    xtrain, xtest, ytrain, ytest = train_test_split(xdata, ydata, test_size = 0.25, stratify = ydata)
    
    return xtrain, xtest, ytrain, ytest