#!/usr/bin/env python
# coding: utf-8

"""
2022-10-04 MY
2022-01-12 MY bin32

env: myGIS or module load geoconda

Features 'VV', 'VH'
From fullArrayIntensities.pkl makes histograms.

INPUTS:
year
startdate
inputdir

OUTPUT:
histograms2020S1-VV_VH.pkl

RUN:

python 02-makeS1Histo-S1.py -y 2020 --startdate 0411 \
    -i /Users/myliheik/Documents/myVEGCOVER/vegcover2023/S1/results-insitu2020/

"""


import argparse
import glob
import textwrap
import time
from pathlib import Path
import os
import numpy as np
import pickle

def save_intensities(filename, arrayvalues):
    with open(filename, 'wb+') as outputfile:
        pickle.dump(arrayvalues, outputfile)
        
def parseDate(startdate, year):
    
    if startdate == '0401':
        date = year + '0401-' + year + '0411'
    elif startdate == '0411':
        date = year + '0411-' + year + '0421'
    elif startdate == '0421':
        date = year + '0421-' + year + '0501'
    elif startdate == '0501':
        date = year + '0501-' + year + '0511'
    elif startdate == '0511':
        date = year + '0511-' + year + '0521'       
    
    return date

def decideBinSeq(outputdirsetti):
    # Read population statistics:
    #filename = os.path.join(outputdirsetti, 'populationStatsS1.pkl')    
    #with open(filename, "rb") as f:
    #    tmp = pickle.load(f)
    #
    #for i in range(4):
    #    print(f'\nFeature:  {list(tmp[i].values())[0]}')
    #    print(f'Percentiles:  {list(tmp[i].values())[5]}') 
    #    print(f'Values:  {list(tmp[i].values())[6]}')
    
    print('We set -33 and -1 values for range.')
    
    vvrange = [-33, -1]
    vhrange = [-33, -1]
    vvvhrange = [1, 17]
    
    return vvrange, vhrange, vvvhrange


def makeHisto(outputdir, bins, vuosi, features):
    #print('\nDecide bin sequences for range...')
    vvrange, vhrange, vvvhrange = decideBinSeq(outputdir)
      
    histlist = []
    nrbins = bins

    maximum = vvrange[1]
    minimum = vvrange[0]
    bin_seqvv = np.linspace(minimum,maximum,nrbins+1)    
    
    maximum = vhrange[1]
    minimum = vhrange[0]
    bin_seqvh = np.linspace(minimum,maximum,nrbins+1)
    
    filename = os.path.join(outputdir, 'fullArrayIntensities' + vuosi + 'S1-' + '_'.join(features) + '.pkl')

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

            if 'vv' in elem:
                bin_seq = bin_seqvv

            elif 'vh' in elem:
                bin_seq = bin_seqvv

            elif 'vvvh' in elem:
                maximum = vvvhrange[1]
                minimum = vvvhrange[0]
                bin_seq = np.linspace(minimum,maximum,nrbins+1)

            else:    
                #print(elem[7], elem[8])
                continue

            #if min(line) >= maximum:
            #    hist = [float(0)]*(nrbins-1); hist.append(float(1)) 
            #elif max(line) <= minimum:
            #    hist = [1]; hist.extend([0]*(nrbins-1))
            #else:
                #density, _ = np.histogram(line, bin_seq, density=False)
                #hist = density / float(density.sum())
            hist, _ = np.histogram(line, bin_seq, density=False)

            myid.extend(hist)
            hist2 = myid
            #print(hist2)
            histlist.append(hist2)
    
    print(f'Saving {len(histlist)} parcels as histograms. Number of null value parcels: {tyhjia}')
    print('Saving histograms into histograms' + vuosi + 'S1-' + '_'.join(features) + '.pkl...')
    save_intensities(os.path.join(outputdir, 'histograms' + vuosi + 'S1-' + '_'.join(features) + '.pkl'), histlist)        



# HERE STARTS MAIN:

def main(args):

    try:
        if not args.year or not args.inputpath:
            raise Exception('Missing year or output directory arguments. Try --help .')

        print(f'\n02-makeS1Histo-S1.py')
        print(f'\nMaking histograms from S1 fullArrays for the year {args.year}.')
   
        t0 = time.time()

        nrbins = 32
    
        out_dir_path = Path(os.path.expanduser(args.inputpath + args.year + args.startdate))
        out_dir_path.mkdir(parents=True, exist_ok=True)
        

        date = parseDate(args.startdate, args.year)
            
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