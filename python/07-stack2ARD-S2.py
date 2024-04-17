"""

23.10.2020 
27.1.2021 updated: replace DOY to order number 1,2,3,4,... during the season. We will need 4 datasets (June, July, August)
19.8.2021 added options 1) to merge given years and data sets, 2) outputdir
1.9.2021 saves in compressed numpy file instead of pickle
15.11.2021 oli törkeä virhe combineAllYears-funktiossa, pitää ajaa uudelleen kaikki
25.11.2021 oli törkeä virhe reshapeAndSave-funktiossa (pivot, reindex, reshape), pitää ajaa uudelleen kaikki
5.11.2022 modified for SISA: farmID -> parcelID
13.1.2023 modified for vegcover2023 -> setti None; farms -> parcels (affects texts only)

Combine annual stack-files into one array stack.

combineAllYears() reads all annuals into one big dataframe.

reshapeAndSave() pivots the dataframe by parcelID and doy, converts to numpy array, fills with na (-> not ragged) and reshapes into 3D. Saves array and parcelIDs into separate files.

RUN: 

python ../python/07-stack2ARD-S2.py -o /Users/myliheik/Documents/myVEGCOVER/vegcover2023/cloudless/fused-insitu -i \ /Users/myliheik/Documents/myVEGCOVER/vegcover2023/cloudless/fused-insitu_annual -y 2020 2021 2022 -r

python ../python/07-stack2ARD-S2.py -o /Users/myliheik/Documents/myVEGCOVER/vegcover2023/cloudless/fused-bigRefe -i \ /Users/myliheik/Documents/myVEGCOVER/vegcover2023/cloudless/fused-bigRefe_annual -y 2020 2021 2022 -r

After this into 08-normalizeS2.py and 09-mergeS1S2.py.


"""
import glob
import os
import pandas as pd
import numpy as np
import pickle

from pathlib import Path

import argparse
import textwrap
from datetime import datetime


###### FUNCTIONS:

def load_intensities(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def save_intensities(filename, arrayvalues):
    with open(filename, 'wb+') as outputfile:
        pickle.dump(arrayvalues, outputfile)

def combineAllYears(data_folder3, setti, years):
    # read files in inputdir:
    s = pd.Series(glob.glob(data_folder3 + '/*.pkl'))

    filepaths = [] 

    for filename in s:
        for keyword1 in years:
            if keyword1 in filename:
                if setti:
                    for keyword2 in setti:
                        if keyword2 in filename:
                            #print(filename)
                            filepaths.append(filename)
                else:
                    filepaths.append(filename)
    # open all chosen years into one dataframe:
    allyears = pd.concat(map(pd.read_pickle, filepaths), sort=False)
    return allyears

def reshapeAndSave(full_array_stack, out_dir_path, outputfile, rank):
    # reshape and save data to 3D:
    print(f"\nLength of the data stack dataframe: {len(full_array_stack)}")

    if rank:
        dateVar = 'doyid'
    else:
        dateVar = 'doy'

    full_array_stack['doyid'] = full_array_stack.groupby(['parcelID', 'band'])['doy'].rank(method="first", ascending=True).astype('int')
    
    final = full_array_stack

    # Kuinka monta havaintoa per tila koko kesältä, mediaani?
    print("How many observations per parcel in one season (median)?: ", float(final[['parcelID', dateVar]].drop_duplicates().groupby(['parcelID']).count().median()))
    # Kuinka monta havaintoa per tila koko kesältä, max?
    print("How many observations per parcel in one season (max)?: ", float(final[['parcelID', dateVar]].drop_duplicates().groupby(['parcelID']).count().max()))
    # Kuinka monta havaintoa per tila koko kesältä, min?
    print("How many observations per parcel in one season (min)?: ", float(final[['parcelID', dateVar]].drop_duplicates().groupby(['parcelID']).count().min()))

    if final[['parcelID', 'band', 'doy']].duplicated(keep = False).any():
        print(f"There are {final[['parcelID', 'band', 'doy']].duplicated(keep = False).sum()} duplicates out of {final.parcelID.nunique()} parcels. We take the first obs. only.")
        final2 = final.drop_duplicates(subset=['parcelID', 'band', 'doy'], keep='first')
        final = final2.copy()
        print(f"Are there duplicates anymore: {final[['parcelID', 'band', 'doy']].duplicated(keep = False).any()}")

    pivoted = final.pivot(index=['parcelID', dateVar], columns='band', values=[*final.columns[final.columns.str.startswith('bin')]])
    m = pd.MultiIndex.from_product([pivoted.index.get_level_values(0).unique(), pivoted.index.get_level_values(1).sort_values().unique()], names=pivoted.index.names)
    pt = pivoted.reindex(m, fill_value = 0)

    farms = pivoted.index.get_level_values(0).nunique()
    doys = len(pivoted.index.get_level_values(1).sort_values().unique())
    bands = 10
    bins = 32
    
    finalfinal = pt.to_numpy().reshape(farms, doys, bins, bands).swapaxes(2,3).reshape(farms,doys,bands*bins)
    
    outputfile2 = 'array_' + outputfile
    fp = os.path.join(out_dir_path, outputfile2)
    
    print(f"Shape of the 3D stack dataframe: {finalfinal.shape}")
    print(f"Output into file: {fp}")
    np.savez_compressed(fp, finalfinal)
    
    # save parcelIDs for later merging with target y:
    parcelIDs = pt.index.get_level_values(0).unique().str.rsplit('_',1).str[0].values
    print(f"\n\nNumber of parcels: {len(parcelIDs)}")
    outputfile2 = 'parcelID_' + outputfile + '.pkl'
    fp = os.path.join(out_dir_path, outputfile2)
    print(f"Output parcelIDs in file: {fp}")
    save_intensities(fp, parcelIDs)
    

    
def main(args):
    
    try:
        if not args.outdir:
            raise Exception('Missing output dir argument. Try --help .')

        print(f'\n\n07-stack2ARD-S2.py')
        print(f'\nInput files in {args.inputdir}')

        # directory for input, i.e. annual results:
        data_folder3 = args.inputdir
        
        # directory for outputs:
        out_dir_path = args.outdir
        Path(out_dir_path).mkdir(parents=True, exist_ok=True)
        
        # years:
        years = args.ylist
        setti = args.setti
        
        # outputfilename:
        #outputfile = '-'.join(setti) + '-' + '-'.join(years) + '.pkl'
        if setti:
            outputfile = '-'.join(setti) + '-' + '-'.join(years)
        else:
            outputfile = '-'.join(years)

                
        print("\nPresuming preprocessing done earlier. If not done previously, please, run with histo2stack.py first!")

        print("\nCombining the years and data sets...")
        allyears = combineAllYears(data_folder3, setti, years)
        reshapeAndSave(allyears, out_dir_path, outputfile, args.rank)
        

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
    # is not true: cannot combine multiple data sets (crops), because parcelID does not hold crop information -> duplicated parcelIDs  
    parser.add_argument('-f', '--setti', action='store', dest='setti',
                         type=str, nargs='*', default=None,
                         help='Name of the data set. Can be also multiple. E.g. -f 1310 1320. Default None.')
    #parser.add_argument('-f', '--setti', 
    #                    type=str,
    #                    default=['1400'],
    #                    help='Name of the data set. E.g. -f 1310.')
    parser.add_argument('-y', '--years', action='store', dest='ylist',
                       type=str, nargs='*', default=['2018', '2019', '2020', '2021', '2022'],
                       help="Optionally e.g. -y 2018 2019, default all")
    
    parser.add_argument('-r', '--rank',
                        help='If saving time series by rank of days.',
                        default=False,
                        action='store_true')
        
    args = parser.parse_args()
    main(args)



