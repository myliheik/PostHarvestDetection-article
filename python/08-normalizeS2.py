"""
MY 2022-10-19, 2023-01-16

Apply l1 normalization to S2 3D ard files.

RUN: 

*** Fusion:


python 08-normalizeS2.py -i /Users/myliheik/Documents/myVEGCOVER/vegcover2023/cloudless/fused-bigRefe/array_2020-2021-2022.npz

python 08-normalizeS2.py -i /Users/myliheik/Documents/myVEGCOVER/vegcover2023/cloudless/fused-insitu/array_2020-2021-2022.npz


*** timeseries:


python 08-normalizeS2.py -i /Users/myliheik/Documents/myVEGCOVER/vegcover2023/cloudless/dataStack-bigRefe/array_2020-2021-2022.npz

python 08-normalizeS2.py -i /Users/myliheik/Documents/myVEGCOVER/vegcover2023/cloudless/dataStack-insitu/array_2020-2021-2022.npz




AFTER:
09-mergeS1S2.py

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

def normalize(input_file):
    # read inputfile:
    data = utils.load_npintensities(input_file)
    normalizer = 'l1'
    dataNorm = utils.normalise3D(data, normalizer)
    
    # back to 3D:
    m,n = dataNorm.shape[:2]
    data3d = dataNorm.reshape(m,n,-1) 
    
    outputfile = input_file.replace('array', 'normalized3D')
    print(f'Data back to 3D and saving into {outputfile} ...')
    np.savez_compressed(outputfile, data3d)

    # is this needed?
    # back to 2D: # in time series case 3D is needed in the next step, so 2D is obsolete
    #outputfile = input_file.replace('array', 'normalized2D')
    #data2d = data3d.reshape(m,-1) 
    #print(f'Data back to 2d and saving into {outputfile} ...')    
    #np.savez_compressed(outputfile, data2d)
    
def main(args):
    
    try:
        if not args.inputfile:
            raise Exception('Missing inputfile. Try --help .')

        print(f'\n\n08-normalizeS2.py')
        print(f'\nInput file in {args.inputfile}')
        
        print("\nNormalizing...")
        
        if os.path.isfile(args.inputfile):
            normalize(args.inputfile)
        else:
            print(f'No such file {args.inputfile} ...')
        

    except Exception as e:
        print('\n\nUnable to read input or write out results. Check prerequisites and see exception output below.')
        parser.print_help()
        raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=textwrap.dedent(__doc__))

    parser.add_argument('-i', '--inputfile',
                        type=str,
                        help='Name of the input directory (where annual histogram dataframes are).',
                        default='.')
    args = parser.parse_args()
    main(args)




