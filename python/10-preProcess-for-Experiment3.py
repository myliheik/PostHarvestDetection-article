"""
2024-02-16 MY

Preprocess and save datasets for experiment 3.

RUN:

# S1S2 3D:
10-preProcess-for-Experiment3.py -i /Users/myliheik/Documents/myVEGCOVER/vegcover2023/cloudless/dataStack-insitu/ard-S1-S2-3D_2020-2021-2022-2023.npz \
-o /Users/myliheik/Documents/myVEGCOVER/vegcover2023/results/experiment3 \
-c 12340

# S2 3D:
10-preProcess-for-Experiment3.py -i /Users/myliheik/Documents/myVEGCOVER/vegcover2023/cloudless/dataStack-insitu/ard-S2-3D_2020-2021-2022-2023.npz \
-o /Users/myliheik/Documents/myVEGCOVER/vegcover2023/results/experiment3 \
-c 12340


# S2 3D, duplicates (due to overlapping tiles) removed:
10-preProcess-for-Experiment3.py -i /Users/myliheik/Documents/myVEGCOVER/vegcover2023/cloudless/ard-S2-3D-noDuplicates_2020-2021-2022-2023.npz \
-o /Users/myliheik/Documents/myVEGCOVER/vegcover2023/results/experiment3 \
-c 12340

# NOTE: hard coded link to reference file in function 'initialization'

"""

import os.path
from pathlib import Path
import time
import argparse
import textwrap
import pandas as pd
import numpy as np
import utils
import utils2



def initialization(inputfile):

    # in situ metat:
    omameta = pd.read_csv('/Users/myliheik/Documents/myVEGCOVER/vegcover2023/metafiles/omameta.csv').drop_duplicates(subset=['parcelID'])
    # bigRefe meta löytyy täältä: 
    #megameta = pd.read_csv('/Users/myliheik/Documents/myVEGCOVER/vegcover2023/metafiles/meta1234VarmatFilteredOutInSitu.csv')    

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


#################################################################################################



def main(args):

    try:
        if not args.inputfile or not args.outputpath:
            raise Exception('Missing input file or output directory argument. Try --help .')

        print(f'\n10-preProcess-for-Experiment3.py')
        print(f'\nPreprocess data for Experiment 3.')
   
        out_dir_path = Path(os.path.expanduser(args.outputpath))
        out_dir_path.mkdir(parents=True, exist_ok=True)
               
        classes = args.classes
        
        head, tail = os.path.split(args.inputfile)
        tunniste = tail.split('.')[0]

        if 'fused' in args.inputfile:
            setti = 'Fusion'
        else:
            setti = 'timeseries'      
          

        df0, metatMeta = initialization(args.inputfile)
        
        if 'noDuplicates' in args.inputfile:
            xtrain, xtest, ytrain, ytest = utils2.munchS2(df0, metatMeta, classes) # no need to drop duplicates)

        elif 'S1' in args.inputfile:
            xtrain, xtest, ytrain, ytest = utils2.munchS1S2(df0, metatMeta, classes)# when  S1 included needs to sweep duplicates into training set

        else:
            xtrain, xtest, ytrain, ytest = utils2.munchS2(df0, metatMeta, classes) # when just S2, no need to drop duplicates)

        print(f'Case {args.classes}:\nLength of the data set: {len(df0)} \n')
        print(f'Class distribution in data set:\n {np.unique(metatMeta["target"], return_counts=True)}')
        print(f'Class distribution in train set:\n {np.unique(ytrain, return_counts=True)}')
        print(f'Class distribution in test set:\n {np.unique(ytest, return_counts=True)}')

        ######################################################################################################################

        # Save datasets:
        xtrainfile = '-'.join(['xtrain', tunniste, classes[0], setti])
        xtestfile = '-'.join(['xtest', tunniste, classes[0], setti])
        ytrainfile = '-'.join(['ytrain', tunniste, classes[0], setti])
        ytestfile = '-'.join(['ytest', tunniste, classes[0], setti])

        print(f'Saving xtrain into {xtrainfile}')
        np.savez_compressed(os.path.join(out_dir_path, xtrainfile), xtrain)
        print(f'Saving xtest into {xtestfile}')
        np.savez_compressed(os.path.join(out_dir_path, xtestfile), xtest)
        print(f'Saving ytrain into {ytrainfile}')
        np.savez_compressed(os.path.join(out_dir_path, ytrainfile), ytrain)
        print(f'Saving ytest into {ytestfile}')
        np.savez_compressed(os.path.join(out_dir_path, ytestfile), ytest)



            
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
                       type=str, nargs='*', default=['1234'],
                       help="Classes to use, e.g. -c 1234 10 0")
        
  
    
    args = parser.parse_args()
    main(args)    
