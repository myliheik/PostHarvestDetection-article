"""
MY 2022-10-19
2022-12-08 revisited time series
2023-01-16 vegcover2023

Merge S1 and S2 index files.


RUN: 

*** mosaics:

python 09-mergeS1S2ind.py -i /Users/myliheik/Documents/myVEGCOVER/vegcover2023/S1/results-bigRefe/combined/ard-20200411_20210411_20220421-l1.csv.gz \
-j /Users/myliheik/Documents/myVEGCOVER/vegcover2023/S2indices/results-bigRefe/combined/ard-20190401_20200401_20210401_20220401-l1.csv.gz

python 09-mergeS1S2ind.py -i /Users/myliheik/Documents/myVEGCOVER/vegcover2023/S1/results-insitu/combined/ard-20200411_20210411_20220421-l1.csv.gz \
-j /Users/myliheik/Documents/myVEGCOVER/vegcover2023/S2indices/results-insitu/combined/ard-20200401_20210401_20220401-l1.csv.gz

AFTER:
16-classifiers
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

def mergeFiles(inputfileS1, inputfileS2):
    
    # output:
    if 'insitu' in inputfileS2:
        out_dir_path = '/Users/myliheik/Documents/myVEGCOVER/vegcover2023/S2indices/dataStack-insitu'
        outputpath = Path(out_dir_path)
        outputpath.mkdir(parents=True, exist_ok=True)
    else:
        out_dir_path = '/Users/myliheik/Documents/myVEGCOVER/vegcover2023/S2indices/dataStack-bigRefe'
        outputpath = Path(out_dir_path)
        outputpath.mkdir(parents=True, exist_ok=True)
        
    # filenames:                      
    filenameS1S22D = 'ard-S1-S2ind-2D'
    columnnameS1S22D = 'columns-S1-S2ind-2D.pkl'
    target = 'parcelID-ard'
    metafileOut = 'parcelID-meta.pkl'

    print('Selecting only parcels in meta files, not to mix train/test!')

    fp = '/Users/myliheik/Documents/myVEGCOVER/vegcover2023/metafiles/omameta.csv'
    metat = pd.read_csv(fp)

    # Read data:
    print(f'Reading S1 file {inputfileS1} ...')
    dfS100 = pd.read_csv(inputfileS1)
    maskS1 = dfS100['parcelID'].isin(metat['parcelID'])
    print(f'Reading S2 file {inputfileS2} ...')
    dfS200 = pd.read_csv(inputfileS2)   
    maskS2 = dfS200['parcelID'].isin(metat['parcelID'])
    
    if 'bigRefe' in inputfileS2:     
        dfS10 = dfS100[~maskS1]
        dfS20 = dfS200[~maskS2]
    else: # in situ
        dfS10 = dfS100[maskS1]
        dfS20 = dfS200[maskS2]        
    
    #print(dfS1.columns)
    print(f'There was {len(dfS100)} parcels in S1 data. Found {len(dfS10)} parcels after merge.')
    print(f'There was {len(dfS200)} parcels in S2 data. Found {len(dfS20)} parcels after merge.')

    dfS = dfS10.merge(dfS20, how = 'inner', on = 'parcelID')

    print(f'There was {len(dfS10)} parcels in S1 data and {len(dfS20)} in S2 data. Found {len(dfS)} parcels after merge.')
        

    #print(len(dfS[dfS.target_x == dfS.target_x]), len(dfS)) # on samat targetit
    print(f'Columns: {dfS.columns}')
    utils.save_intensities(os.path.join(outputpath, columnnameS1S22D), dfS.columns)        

    xdata = dfS.loc[:, dfS.columns.str.contains('bin')].values
    ydata = dfS['target_x'].values    
    meta = dfS.loc[:, ~dfS.columns.str.contains('bin')]
    
    print('These features:\n')
    print(dfS.loc[:, dfS.columns.str.contains('bin')].columns.values)

    ######### Saving:

    print(f'Saving 2D S1&S2ind ard \n having shape of {xdata.shape}\n into {os.path.join(outputpath, filenameS1S22D)} ')
    np.savez_compressed(os.path.join(outputpath, filenameS1S22D), xdata) 

    print(f'Saving 2D S1&S2ind y \n having shape of {ydata.shape}\n into {os.path.join(outputpath, target)} ')
    np.savez_compressed(os.path.join(outputpath, target), ydata) 

    # meta file saved:
    print(f"Saving S1&S2 metafile  \n having shape of {meta.shape}\n into {os.path.join(outputpath, metafileOut)}")
    utils.save_intensities(os.path.join(outputpath, metafileOut), meta)        
        

        
def main(args):
    
    try:
        if not args.inputfileS2:
            raise Exception('Missing S2 index inputfile. Try --help .')

        print(f'\n\n09-mergeS1S2ind.py')
        #print(f'\nInput file in {args.inputfile1}')
        
        print("\n ...")
        
        if os.path.isfile(args.inputfileS2):
            mergeFiles(args.inputfileS1, args.inputfileS2)
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
                       help="Input file for normalized S2 data.")  
    parser.add_argument('-j', '--inputfileS2',
                        type=str,
                        help='Input file for normalized S2 index data.')
    #parser.add_argument('-c', '--closelyWatched',
    #                    help="Select only closelyWatched parcels. Applies only when using S1 data and to test set data only.",
    #                    default=False,
    #                    action='store_true') 
    args = parser.parse_args()
    main(args)




