"""
MY 2022-10-19
2022-12-08 revisited time series
2023-01-16 vegcover2023
2023-12-08 added: drop S2 duplicates (due to overlapping orbits) for comparison purposes, option: -d
            but not for time series!

Merge S1 and S2 files.


RUN: 

*** Fusion:

python 09-mergeS1S2.py -i /Users/myliheik/Documents/myVEGCOVER/vegcover2023/S1/results-bigRefe/combined/ard-20200411_20210411_20220421-l1.csv.gz \
-j /Users/myliheik/Documents/myVEGCOVER/vegcover2023/cloudless/fused-bigRefe/normalized3D_2020-2021-2022.npz

python 09-mergeS1S2.py -i /Users/myliheik/Documents/myVEGCOVER/vegcover2023/S1/results-insitu/combined/ard-20200411_20210411_20220421-l1.csv.gz \
-j /Users/myliheik/Documents/myVEGCOVER/vegcover2023/cloudless/fused-insitu/normalized3D_2020-2021-2022.npz

*** timeseries:

python 09-mergeS1S2.py -i /Users/myliheik/Documents/myVEGCOVER/vegcover2023/S1/results-bigRefe/combined/ard-20200411_20210411_20220421-l1.csv.gz \
-j /Users/myliheik/Documents/myVEGCOVER/vegcover2023/cloudless/dataStack-bigRefe/normalized3D_2020-2021-2022.npz

python 09-mergeS1S2.py -i /Users/myliheik/Documents/myVEGCOVER/vegcover2023/S1/results-insitu/combined/ard-20200411_20210411_20220421-l1.csv.gz \
-j /Users/myliheik/Documents/myVEGCOVER/vegcover2023/cloudless/dataStack-insitu/normalized3D_2020-2021-2022.npz


AFTER:
17-classifiers
"""

import os
import pandas as pd
import numpy as np
import pickle
import utils

from pathlib import Path

import argparse
import textwrap



###### FUNCTIONS:

def mergeFiles(inputfileS1, inputfileS2, dropDuplicates):
    
    # filenames:        
    metafile0 = inputfileS2.replace('normalized3D', 'parcelID')
    metafile = metafile0.replace('npz', 'pkl')
    
    if dropDuplicates:   # not for time series            
        filenameS22D = inputfileS2.replace('normalized3D', 'ard-S2-2D-noDuplicates')
        filenameS13D = inputfileS2.replace('normalized3D', 'ard-S1-3D-noDuplicates')
        #filenameS1S23Dtimeseries = inputfileS2.replace('normalized3D', 'ard-S1-S2-3D-noDuplicates')
        #filenameS23Dtimeseries = inputfileS2.replace('normalized3D', 'ard-S2-3D-noDuplicates')
        filenameS1S22Dfusion = inputfileS2.replace('normalized3D', 'ard-S1-S2-2D-noDuplicates')
        metafileOut = metafile.replace('parcelID', 'parcelID-ard-noDuplicates')
    else:
        filenameS22D = inputfileS2.replace('normalized3D', 'ard-S2-2D')
        filenameS13D = inputfileS2.replace('normalized3D', 'ard-S1-3D')
        filenameS1S23Dtimeseries = inputfileS2.replace('normalized3D', 'ard-S1-S2-3D')
        filenameS23Dtimeseries = inputfileS2.replace('normalized3D', 'ard-S2-3D')
        filenameS1S22Dfusion = inputfileS2.replace('normalized3D', 'ard-S1-S2-2D')
        metafileOut = metafile.replace('parcelID', 'parcelID-ard')

    
    # Read S1 data:
    print(f'Reading S1 file {inputfileS1} ...')
    dfS10 = pd.read_csv(inputfileS1)
    print(dfS10.head(1))

    
    if 'bigRefe' in inputfileS2: 
        # big refe meta:
        fp  = '/Users/myliheik/Documents/myVEGCOVER/vegcover2023/metafiles/meta1234VarmatFilteredOutInSitu.csv'
    else: # in situ meta
        fp = '/Users/myliheik/Documents/myVEGCOVER/vegcover2023/metafiles/omameta.csv'
    metat = pd.read_csv(fp)


    dfS1 = dfS10.merge(metat, how = 'inner')
    #print(dfS1.columns)
    print('Selecting only parcels having in situ observed!')
    print(f'There was {len(dfS10)} parcels in S1 data. Found {len(dfS1)} parcels after merge.')
        

    # 2D merge vain jos Fusion!
    if 'fused' in inputfileS2:
        print('Fusion data')


        # S2 array data:
        arrayS2 = utils.load_npintensities(inputfileS2)
        # parcelIDs:
        dfS2 = pd.DataFrame(utils._load_intensities(metafile), columns = ['parcelID'])
        # make sure index follows:
        dfS2['S2_index'] = dfS2.index

        # katsotaan S2 metafile, onko siinä useita vuosia?
        years = pd.unique(dfS2['parcelID'].str.split('_', expand = True)[0]).tolist()
        print(f'S2 data includes years {", ".join(years)}.')
        # katsotaan S1 dataframe, onko siinä useita vuosia?
        yearsS1 = pd.unique(dfS1['parcelID'].str.split('_', expand = True)[0]).tolist()
        print(f'S1 data includes years {", ".join(yearsS1)}.')
        
        if len(yearsS1) == len(years):
            print(f'OK, the numbers of years match.')
        else:
            print(f'Note! S1 years ({len(yearsS1)}) and S2 years dont match ({len(years)}).')
            
            
        if dropDuplicates:
            dfS22 = dfS2.drop_duplicates('parcelID')
        else: 
            dfS22 = dfS2
            

        # merge in S2 parcels with S1, take all S1:
        dfmerged = dfS1.merge(dfS22, how = 'inner', on = 'parcelID')
        # make sure the order is S2 data order:
        dfmerged2 = dfmerged.sort_values(by=['S2_index'])
        
        print(dfS2.shape, dfmerged.shape, dfmerged2.shape)
        #print(dfmerged2.head())
        
        print(f'Initial shapes S2: {arrayS2.shape}, S1: {dfS1.shape}, common unique parcels found: {len(pd.unique(dfS2["parcelID"]))}.')
        #print('There are duplicates because of multiple overlapping S2 tiles')
        
        # mask S2 array by the common parcels:
        maskS20 = dfS2['parcelID'].isin(dfmerged2['parcelID'])
        
        if dropDuplicates:
            # mask duplicates from S2:
            maskDuplicates = dfS2['parcelID'].duplicated()
            maskS2 = (~maskDuplicates & maskS20)
        else:
            maskS2 = maskS20
            
        #print(dfS2.shape, dfS1.shape, dfmerged.shape, dfmerged2.shape, arrayS2.shape, len(maskS2))
        if len(maskS2) == len(arrayS2):
            xdata = arrayS2[maskS2,:,:]  
            ydata = dfS2['parcelID'][maskS2]
        else:
            print('Mismatch of mask and S2 array data!')
            
        #print(xdata.shape, ydata.shape, len(maskS2))
        
        # make 3D Fusion S1 ard:
        
        vv = dfmerged2.loc[:, dfmerged2.columns.str.contains('vv')].values
        vh = dfmerged2.loc[:, dfmerged2.columns.str.contains('vh')].values
        S1array = np.dstack((vv,vh))
        S1array2 = np.swapaxes(S1array, 1, 2)
        S1array3 = np.expand_dims(S1array2, axis=1)
        # back to 3D:
        m,n = S1array3.shape[:2]
        S1array4 = S1array3.reshape(m,n,-1) 
        print(S1array.shape, S1array4.shape, xdata.shape)

        ## make 3D Fusion S1&S2 ard:
        #dfmerged2DFiltered = dfmerged2D.loc[:, (dfmerged2D.columns.str.startswith("bin"))]
        #m,n = dfmerged2DFiltered.shape[:2]
        #dfmerged3D = dfmerged2DFiltered.values.reshape(m, -1, 16)
        # pitää muokata 3D 4D:n kautta:
        #dfmerged4D = np.expand_dims(dfmerged3D, axis=1)
        ## back to 3D:
        #m,n = dfmerged4D.shape[:2]
        #dfmerged3D2 = dfmerged4D.reshape(m,n,-1)  
        
        
        #print(f'Saving 3D S1&S2 ard into {filenameS1S23D} \n Shape of {dfmerged3D2.shape}')
        #np.savez_compressed(filenameS1S23D, dfmerged3D2)  
        
        # meta file saved:
        #metafileOut = metafile.replace('parcelID', 'parcelID-ard-S1-S2-3D')
        #print(f"Saving S1&S2 time series metafile into {metafileOut} \n Shape of {dfmerged2D['parcelID'].shape}")
        #utils.save_intensities(metafileOut, dfmerged2D['parcelID'])
        
        # stack S1 & S2 in 3D:
        print(f'Stack S1 & S2 in 3D. Shape of S1array4: {S1array4.shape} \n Shape of xdata: {xdata.shape}')

        S1S2data3D = np.dstack((S1array4, xdata))
        S1S2data2D = S1S2data3D.reshape(S1S2data3D.shape[0], -1)

        print(f'S1 shape in 3D: {S1array2.shape} (obs, bands, features), with time dimension: {S1array3.shape}, reshaped: {S1array4.shape}, stacked with S2 array: {S1S2data3D.shape}')
        
        
        ######### Saving:
        print(f'Saving 3D S1&S2 ard into {filenameS1S23Dtimeseries} \n Shape of {S1S2data3D.shape}')
        np.savez_compressed(filenameS1S23Dtimeseries, S1S2data3D) 

        print(f'\nSaving 3D S1 fusion ard  \n having shape of {S1array2.shape}\n into {filenameS13D}')
        np.savez_compressed(filenameS13D, S1array2)

        print(f'Saving 2D S1&S2 ard \n having shape of {S1S2data2D.shape}\n into {filenameS1S22Dfusion} ')
        np.savez_compressed(filenameS1S22Dfusion, S1S2data2D) 

        
        # meta file saved:
        print(f"Saving S1&S2 metafile  \n having shape of {ydata.shape}\n into {metafileOut}")
        utils.save_intensities(metafileOut, ydata)
        
        # save also separately S2 and S1:
        #print(f"Saving S1 in 2D" ) # TODO! NB: S1array on ok, mutta pitää poistaa duplikaatit
        #S1array4 # S1 in 2D only
        
        #print(f'Saving 3D S2 only ard into {filenameS23Dtimeseries} \n Shape of {xdata.shape}')
        #np.savez_compressed(filenameS23Dtimeseries, xdata) # S2 in 3D
        
        # varmat filtteröinti ei ulotu tähän, joten ei tehdä.
        #xdata2D = arrayS2.reshape(arrayS2.shape[0], -1) # S2 in 2D
        #print(f'Saving 2D S2 only ard having shape of {xdata2D.shape}\n into {filenameS22D} \n')
        #np.savez_compressed(filenameS22D, xdata2D) # S2 in 2D       

        
        
    else:
        print('timeseries')
        # array data:
        arrayS2 = utils.load_npintensities(inputfileS2)
        # parcelIDs:
        dfS2 = pd.DataFrame(utils._load_intensities(metafile), columns = ['parcelID'])
        # make sure index follows:
        dfS2['S2_index'] = dfS2.index

            
        # katsotaan S2 metafile, onko siinä useita vuosia?
        years = pd.unique(dfS2['parcelID'].str.split('_', expand = True)[0]).tolist()
        print(f'S2 data includes years {", ".join(years)}.')

        # katsotaan S1 dataframe, onko siinä useita vuosia?
        yearsS1 = pd.unique(dfS1['parcelID'].str.split('_', expand = True)[0]).tolist()
        print(f'S1 data includes years {", ".join(yearsS1)}.')
        
        if len(yearsS1) == len(years):
            print(f'OK, the numbers of years match.')
        else:
            print(f'Note! S1 years ({len(yearsS1)}) and S2 years dont match ({len(years)}).')

        #if dropDuplicates:
        #    dfS22 = dfS2.drop_duplicates('parcelID')
        #else: 
        #    dfS22 = dfS2
            
        # merge in S2 parcels with S1, take all S1:
        #dfmerged = dfS1.merge(dfS2, how = 'left', on = 'parcelID')
        dfmerged = dfS1.merge(dfS2, how = 'inner', on = 'parcelID')
        # make sure the order is S2 data order:
        dfmerged2 = dfmerged.sort_values(by=['S2_index'])
        
        timepoints = arrayS2.shape[1]
        
        print(f'Initial shapes S2: {arrayS2.shape}, S1: {dfS1.shape}, common unique parcels found: {len(pd.unique(dfmerged2["parcelID"]))}.')
        print('There are duplicatesin S2 data because of multiple overlapping S2 tiles.')

        # mask S2 array by the common parcels:
        maskS2 = dfS2['parcelID'].isin(dfmerged2['parcelID'])
        #print(dfS2.shape, dfS1.shape, dfmerged.shape, dfmerged2.shape, arrayS2.shape, len(maskS2))
        if len(maskS2) == len(arrayS2):
            xdata = arrayS2[maskS2,:,:]  
            ydata = dfS2['parcelID'][maskS2]
        else:
            print('Mismatch of mask and S2 array data!')
      
        #print(xdata.shape, ydata.shape, len(maskS2))
        # extract S1 features:
        vv = dfmerged2.loc[:, dfmerged2.columns.str.contains('vv')].values
        vh = dfmerged2.loc[:, dfmerged2.columns.str.contains('vh')].values

        S1array = np.dstack((vv,vh)) # obs, features, bands
        S1array2 = np.swapaxes(S1array, 1, 2) # obs, bands, features
        S1array3 = np.expand_dims(S1array2, axis=1) # obs, time, bands, features
        # back to 3D:
        m,n = S1array3.shape[:2]
        S1array4 = S1array3.reshape(m,n,-1) # obs, time, features x bands
        #print(S1array.shape, S1array2.shape, S1array3.shape, S1array4.shape)
        
        
        print(S1array4.shape, m, timepoints-1)
        print('Zero-pad S1 data (only 1 timepoint) to have the same number of time points are S2 data')
        #S1array4padded = np.hstack((np.zeros((m, timepoints-1, S1array4.shape[2])), S1array4)) # kokeilin, mutta ei parempi
        S1array4padded = np.hstack((S1array4, np.zeros((m, timepoints-1, S1array4.shape[2]))))
        print(f'Stack S1 & S2 in 3D. Shape of S1aarray4padded: {S1array4padded.shape} \n Shape of xdata: {xdata.shape}')
        S1S2data3D = np.dstack((S1array4padded, xdata))

        print(f'S1 shape in 3D: {S1array2.shape} (obs, bands, features), reshaped, with time dimension: {S1array4.shape}, zero-padded: {S1array4padded.shape}, stacked with S2 array: {S1S2data3D.shape}\n\n')
        
        
        ######### Saving:
        print(f'Saving 3D S1&S2 time series ard into {filenameS1S23Dtimeseries} \n Shape of {S1S2data3D.shape}')
        np.savez_compressed(filenameS1S23Dtimeseries, S1S2data3D) 

        # meta file saved:

        print(f"Saving S1&S2 time series metafile into {metafileOut} \n Shape of {ydata.shape}")
        utils.save_intensities(metafileOut, ydata)
        
        # save also separately S2 and S1:
        #print(f"Saving S1 in 2D" ) # TODO! NB: S1array on ok, mutta pitää poistaa duplikaatit
        #S1array4 # S1 in 2D only
        
        print(f'Saving 3D S2 only time series ard into {filenameS23Dtimeseries} \n Shape of {xdata.shape}')
        np.savez_compressed(filenameS23Dtimeseries, xdata) # S2 in 3D
        
        # varmat filtteröinti ei ulotu tähän, joten ei tehdä.
        #xdata2D = arrayS2.reshape(arrayS2.shape[0], -1) # S2 in 2D
        #print(f'Saving 2D S2 only ard into {filenameS22D} \n Shape of {xdata2D.shape}')
        #np.savez_compressed(filenameS22D, xdata2D) # S2 in 2D       
        
        

        
def main(args):
    
    try:
        if not args.inputfileS2:
            raise Exception('Missing S2 inputfile. Try --help .')

        print(f'\n\n09-mergeS1S2.py')
        #print(f'\nInput file in {args.inputfile1}')
        
        print("\n ...")
        
        if os.path.isfile(args.inputfileS2):
            mergeFiles(args.inputfileS1, args.inputfileS2, args.dropDuplicates)
        else:
            print(f'No such file {args.inputfileS2} ...')
        

    except Exception as e:
        print('\n\nUnable to read input or write out results. Check prerequisites and see exception output below.')
        parser.print_help()
        raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=textwrap.dedent(__doc__))
    
    parser.add_argument('-i', '--inputfileS1',
                       type=str,
                       help="Input file for normalized S1 data.")  
    parser.add_argument('-j', '--inputfileS2',
                        type=str,
                        help='Input file for normalized S2 data.')
    parser.add_argument('-d', '--dropDuplicates',
                        help="Drop S2 duplicates.",
                        default=False,
                        action='store_true') 
    args = parser.parse_args()
    main(args)




