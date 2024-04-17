"""
MY 2021-08-24, 2021-11-10, 2023-01-11

Read all pixel values and draw histograms. Took 6mins. Allocated --mem 76000 (maximum). Next time run in sbatch.

RUN:

# cloud masked data:
python 03D-drawS2BandDistributions.py -i /scratch/project_2002224/vegcover2023/cloudless/resultsAllPixels/ \
-o /scratch/project_2002224/vegcover2023/img/ \
-y 2020

WHERE:
-i: input directory 
-o: filename of the output image
-y: year (optional)
-d: if readInPixelValues alredy done, add option -d, so that preprocessed data is loaded and we go straight to plotting.


"""
import os
import seaborn as sns
import re
import time

import pandas as pd
from matplotlib import pyplot as plt 
import numpy as np

from glob import glob 
from pathlib import Path 

import textwrap
import argparse

sns.set_style("whitegrid") # jos haluaa valkoiset taustat

colors = ["skyblue", "green", "red",[0.9312692223325372, 0.8201921796082118, 0.7971480974663592],
           [0.7840440880599453, 0.5292660544265891, 0.6200568926941761],
  [0.402075529973261, 0.23451699199015608, 0.4263168000834109],
         [0.8, 0.47058823529411764, 0.7372549019607844], # B08
  [0.1750865648952205, 0.11840023306916837, 0.24215989137836502],
  [1.0, 0.4980392156862745, 0.054901960784313725],
         [0.9254901960784314, 0.8823529411764706, 0.2]]

def readInPixelValues(filedir: str, features, year):

    data = []

    for band in features:
        if year:
            filename = os.path.join(filedir, 'train' + year + '_' + band + '.csv')
            filename2 = os.path.join(filedir, 'percentiles_' + year + '_' + band + '.csv')
            filename3 = os.path.join(filedir, 'allPixelvalues_' + year + '_' + band)
        else:
            filename = os.path.join(filedir, 'train-' + band + '.csv')       
            filename2 = os.path.join(filedir, 'percentiles_' + band + '.csv')
            filename3 = os.path.join(filedir, 'allPixelvalues_' + band)

        #print('\nCalculate histograms...')
    
        print(f'Read {filename}...')        

        #bigArray = np.array([])
        bigArray = []
        with open(filename) as f:
            for line in f:
                values = line.partition(",")[-1]
                myArray = np.fromstring(values, dtype=float, sep=',')
                #bigArray = np.concatenate([bigArray, myArray])
                bigArray.append(myArray)
        
        bigArrayFlat = np.concatenate(bigArray)
        print(f'Length of band pixels: {len(bigArrayFlat)}.')
        
        print(f'Saving band pixel values into {filename3}...')
        np.savez_compressed(filename3, bigArrayFlat)
        
        print(f'Calculate percentiles and save into {filename2}...')
        p = np.linspace(0, 100, 100)
        tmp = np.percentile(bigArrayFlat,p)
        # Save Numpy array to csv
        np.savetxt(filename2, [tmp], delimiter=',', fmt='%d')

        #print(f'\nLet\'s take a sample of {filename}. Sample size of {samplesize} ({samplesize/len(tmp)}%).')
        #tmp2 = np.random.choice(tmp, size = samplesize, replace = False)
        d = {'x': bigArrayFlat, 'Band': band}         
        df = pd.DataFrame(data = d)
        
        data.append(df)
        
    return data

def readInData(filedir: str, features, year):
    data = []
    
    for band in features:
        if year:
            filename3 = os.path.join(filedir, 'allPixelvalues_' + year + '_' + band + '.npz')
        else:
            filename3 = os.path.join(filedir, 'allPixelvalues_' + band + '.npz')
        
        print(f'Read {filename3}...')  
        
        ss = np.load(filename3)['arr_0']
        d = {'x': ss, 'Band': band}         
        df = pd.DataFrame(data = d)
        print(f'Band {band} has {ss.shape} pixels.\n')            
        data.append(df)

    return data

def drawPlots(data, imgdir: str, features, year):
    t = time.localtime()
    timeString  = time.strftime("%Y-%m-%d", t)
    
    # Plot 1:
    if year:
        imgfile2 = os.path.join(imgdir, 'histograms-' + year + '-cloudmasked-' + timeString + '.png')
        imgfile3 = os.path.join(imgdir, 'histograms-' + year + '-cloudmasked-' + timeString + '.eps')
    else:
        imgfile2 = os.path.join(imgdir, 'histograms-cloudmasked-' + timeString + '.png')
        imgfile3 = os.path.join(imgdir, 'histograms-cloudmasked-' + timeString + '.eps')
        
    print(f'Saving plot 1 to file: {imgfile2}')
    
    features2 = ['B02 Blue', 'B03 Green', 'B04 Red', 'B05 Vegetation\nred edge', 'B06 Vegetation\nred edge', 'B07 Vegetation\nred edge', 'B08 NIR', 'B8A Narrow NIR', 'B11 SWIR', 'B12 SWIR']
    
    fig, ax = plt.subplots()
    for i in range(len(features)):
        sns.histplot(data=data[i], x="x", bins = 1000, binrange = [1,6000], color=colors[i], label=features2[i], stat = 'count', multiple="dodge")
    plt.ylim(0, 600000)
    #plt.yticks(np.arange(0, 5000000, step=500000), ['0', '', '1000000', '', '2000000', '', '3000000', '', '4000000', ''], fontsize=10) 
    plt.xticks(np.arange(0, 6000, step=500), ['0', '', '1000', '', '2000', '', '3000', '', '4000', '', '5000', ''], fontsize=10) 
    #plt.xticklabels(['0', '', '1000', '', '2000', '', '3000', '', '4000', '', '5000', ''],fontsize=10)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    #figure = plt.gcf()
    #figure.set_size_inches(5, 5)
    #figure.set_dpi(300)

    plt.legend(loc="upper right")
    ax.set_xlabel('Reflectance')
    ax.set_ylabel('Count')
    plt.grid(True, alpha=.5)

    plt.savefig(imgfile2, format='png', dpi=300)
    plt.savefig(imgfile3, format='eps', dpi=300)

# HERE STARTS MAIN:

def main(args):

    try:
        if not args.inputdir: #or not args.feature:
            raise Exception('Missing input directory arguments. Try --help .')

        print(f'\n03D-drawS2BandDistributions.py')
        print(f'\nDraw histograms for features (S2 bands).')
        
        out_dir_path = Path(os.path.expanduser(args.outputdir))
        out_dir_path.mkdir(parents=True, exist_ok=True)

   
        features = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
            
        if not args.draw:            
            data = readInPixelValues(args.inputdir, features, args.year)
            drawPlots(data, out_dir_path, features, args.year)
        if args.draw:
            data = readInData(args.inputdir, features, args.year)
            drawPlots(data, out_dir_path, features, args.year)


        
        print(f'\nDone.')
            
    except Exception as e:
        print('\n\nUnable to read input or write out results. Check prerequisites and see exception output below.')
        parser.print_help()
        raise e
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=textwrap.dedent(__doc__))

    parser.add_argument('-i', '--inputdir',
                        type = str,
                        help = 'Input directory.')
    # tämä ei toiminut:
    #parser.add_argument('-f', '--feature', action='store', dest='alist',
    #                   type=str, nargs='*', default=['VV', 'VH', 'VVVH'],
    #                   help="Optionally e.g. -f VV VH, default all")

    parser.add_argument('-o', '--outputdir',
                        type = str,
                        help = 'Output directory for images.')
    
    parser.add_argument('-y', '--year',
                        type = str,
                        help = 'Year. Optional.',
                        default = None)
    parser.add_argument('-d', '--draw', # if readInPixelValues alredy done
                        help='Load preprocessed band data from directory and draw.',
                        default=False,
                        action='store_true')
        
    parser.add_argument('--debug',
                        help='Verbose output for debugging.',
                        action='store_true')

    args = parser.parse_args()
    main(args)
