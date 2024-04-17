#!/usr/bin/env python
# coding: utf-8

"""
2024-01-01 MY

env: myGIS or module load geoconda

Features 'COH12VV', 'COH12VH'
From fullArrayIntensities.pkl makes histograms.

INPUTS:
year
startdate
inputdir

OUTPUT:
histograms2023InSAR-VV_VH.pkl

RUN:

python 02-makeInSARHisto.py -y 2023 --startdate 0408 \
    -i /Users/myliheik/Documents/myVEGCOVER/vegcover2023/InSAR/results-insitu2020/

"""


import argparse
import glob
import textwrap
import time
from pathlib import Path
import os
import numpy as np
import pickle


def save_COH12(filename, arrayvalues):
    with open(filename, 'wb+') as outputfile:
        pickle.dump(arrayvalues, outputfile)

def makeHisto(outputdir, bins, vuosi, features):
    #print('\nDecide bin sequences for range...')
    vvrange = [0, 1]
    vhrange = [0, 1]
      
    histlist = []
    nrbins = bins

    maximum = vvrange[1]
    minimum = vvrange[0]
    bin_seq = np.linspace(minimum,maximum,nrbins+1)    
    
    filename = os.path.join(outputdir, 'fullArrayCOH12' + vuosi + 'InSAR-' + '_'.join(features) + '.pkl')

    print('\nCalculate histograms...')

    with open(filename, "rb") as f:
        tmp = pickle.load(f)
        tyhjia = 0
        for elem in tmp:
            myid = elem[0:7] # 11 ; miksei 7->6, kunte S2index prosessoinnissa?
            line = elem[7:] # 11
            #inx = elem[10] 

            if not line:
                tyhjia = tyhjia + 1
                continue

            # pick these: 'parcelID', 'target', 'split', 'PINTAALA', 'mean', 'median', 'startdate', 'date', 'feature'
            # # 'parcelID', 'target', 'split', 'PINTAALA', 'startdate', 'date', 'feature', 'mean', 'median'
            #alku = myid[0:2]
            #alku.extend(myid[3:5])
            #alku.extend(myid[6:11])
            #myid = alku

            hist, _ = np.histogram(line, bin_seq, density=False)

            myid.extend(hist)
            hist2 = myid
            #print(hist2)
            histlist.append(hist2)
    
    print(f'Saving {len(histlist)} parcels as histograms. Number of null value parcels: {tyhjia}')
    print('Saving histograms into histograms' + vuosi + 'InSAR-' + '_'.join(features) + '.pkl...')
    save_COH12(os.path.join(outputdir, 'histograms' + vuosi + 'InSAR-' + '_'.join(features) + '.pkl'), histlist)        



# HERE STARTS MAIN:

def main(args):

    try:
        if not args.year or not args.inputpath:
            raise Exception('Missing year or output directory arguments. Try --help .')

        print(f'\n02-makeInSARHisto.py')
        print(f'\nMaking histograms from InSAR fullArrays for the year {args.year}.')
   
        t0 = time.time()

        nrbins = 32
    
        out_dir_path = Path(os.path.expanduser(args.inputpath + args.year + args.startdate))
        out_dir_path.mkdir(parents=True, exist_ok=True)
            
        if args.make_histograms:
            makeHisto(out_dir_path, nrbins, args.year, args.alist)

        t1 = time.time()
        total = t1-t0

        print("Everything done, took: " + str(round(total, 0))+"s")
        
        print(f'\nDone.')
            
    except Exception as e:
        print('\n\nUnable to read input or write out results. Check prerequisites and see exception output below.')
        parser.print_help()
        raise e
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=textwrap.dedent(__doc__))

    parser.add_argument('-y', '--year',
                        type = str,
                        help = 'Year.')
    parser.add_argument('-d', '--startdate',
                        type = str,
                        help = 'Starting date of the mosaic. E.g. 0411')
    parser.add_argument('-f', '--features', action='store', dest='alist',
                       type=str, nargs='*', default=['VV', 'VH'],
                       help="Optionally e.g. -f VV VH, default all")  
    parser.add_argument('-i', '--inputpath',
                        type=str,
                        help='Directory for fullArray files.',
                        default='.')
    parser.add_argument('-m', '--make_histograms',
                        help="Extract histograms of (multi)parcel's pixel profile.",
                        default=True,
                        action='store_true')   
    parser.add_argument('--debug',
                        help='Verbose output for debugging.',
                        action='store_true')

    args = parser.parse_args()
    main(args)