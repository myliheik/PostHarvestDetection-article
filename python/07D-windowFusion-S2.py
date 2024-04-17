"""
MY 2022-03-24
2023-01-13 for vegcover2023

Apply to all annual stack-files: add/sum of duplicates per window, i.e. merge all observations per window per parcel into one.

Window is not set, simply makes groupby parcel and band. For several windows, give window data as input and combine then in 07-stack2ARD.py

RUN: 

python ../python/07D-windowFusion-S2.py -i /Users/myliheik/Documents/myVEGCOVER/vegcover2023/cloudless/dataStack-insitu_annual -o /Users/myliheik/Documents/myVEGCOVER/vegcover2023/cloudless/fused-insitu_annual


Before this 06-histo2stack.py, after this 07-stack2ARD.py.

"""
import glob
import os
import pandas as pd
import numpy as np
import pickle
import utils

from pathlib import Path

import argparse
import textwrap



###### FUNCTIONS:

def combineAllDOYs(data_folder, out_dir_path):
    # read files in inputdir:
    s = pd.Series(glob.glob(data_folder + '/*.pkl'))

    for filename in s:
        df = utils._load_intensities(filename)
        
        #df2 = df.groupby(['parcelID', 'band', 'doy']).aggregate(np.sum).reset_index() # doy fusion
        df2 = df.groupby(['parcelID', 'band']).aggregate(np.sum).reset_index()
        #print(df2.head())
        filename2 = os.path.join(out_dir_path, filename.split('/')[-1])

        print(f"Saving {filename} to file: {filename2}")
        utils.save_intensities(filename2, df2)
    

    
def main(args):
    
    try:
        if not args.inputdir or not args.outdir:
            raise Exception('Missing input or output dir. Try --help .')

        print(f'\n\n07D-windowFusion-S2.py')
        print(f'\nInput files in {args.inputdir}')

        # directory for input, i.e. annual results:
        data_folder = args.inputdir
        
        # directory for outputs:
        out_dir_path = args.outdir
        Path(out_dir_path).mkdir(parents=True, exist_ok=True)

        print("\nCombining the doys within fused time window (e.g. 11-days)...")
        combineAllDOYs(data_folder, out_dir_path)
        

    except Exception as e:
        print('\n\nUnable to read input or write out results. Check prerequisites and see exception output below.')
        parser.print_help()
        raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=textwrap.dedent(__doc__))

    parser.add_argument('-i', '--inputdir',
                        type=str,
                        help='Name of the input directory (where annual histogram dataframes are).',
                        default='.')
    parser.add_argument('-o', '--outdir',
                        type=str,
                        help='Name of the output directory.',
                        default='.')
    args = parser.parse_args()
    main(args)




