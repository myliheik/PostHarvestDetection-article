#!/usr/bin/env python
# coding: utf-8

"""
2020-09-16 MY
2021-03-09 MY
2023-02-08 MY
env: myGIS

From Sentinel-2 mosaics to analysis ready data with rasterstats. Indices 'NDBI', 'NDMI', 'NDTI' and 'NDVI'.

@PUHTI

INPUTS:
i: input shapefile of parcels
y: year
o: outputdir
d: start date

binsize is fixed to 32.

OUTPUT:
annual range files + plots of distributions
histograms per parcel, .pkl

- ensin vivulla --extractIntensities saadaan intensiteettitiedostot, jokaiselta vuodelta ('fullArrayIntensities' + vuosi + '.pkl')
	- tallentaa myös 128 bin histogramin kaikista pikseleistä per setti ( populationNDMIIntensities*.pkl)
- lisäksi saadaan koko populaation jakaumasta statiikkaa ('populationStats.pkl')
	- ja kuvia 
- kaiken tämän jälkeen aja vivulla --make_histograms, joka tekee histogrammit per par (histograms.pkl)

Next, run makeARD.py (2D or 3D).

RUN @PUHTI:

for vuosi in 2020 2021 2022 2023; do

 srun python /projappl/project_2002224/vegcover2023/python/01-makeS2IndexHisto.py -y $vuosi -d 0401\
    -i /projappl/project_2002224/vegcover2023/shpfiles/insitu/extendedReferences$vuosi-AOI-sampled-buffered.shp \
    -o /scratch/project_2002224/vegcover2023/S2indices/results-insitu \
    --extractIntensities

 srun python /projappl/project_2002224/vegcover2023/python/01-makeS2IndexHisto.py -y $vuosi -d 0401\
    -o /scratch/project_2002224/vegcover2023/S2indices/results-insitu \
    --plotPopulations

 srun python /projappl/project_2002224/vegcover2023/python/01-makeS2IndexHisto.py -y $vuosi -d 0401\
    -o /scratch/project_2002224/vegcover2023/S2indices/results-insitu \
    --make_histograms

    done

"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd
from pathlib import Path
import pickle

from rasterstats import zonal_stats
import seaborn as sns
import textwrap

import time

# PARAMETERS:
sns.set_style('darkgrid')
nrbins = 16

colors = ["black", "goldenrod" , "saddlebrown", "gold", "skyblue", "yellowgreen", "darkseagreen", "darkgoldenrod"]


# FUNCTIONS:

def plot_histogram(values, title:str, save_in=None):
    g = sns.distplot(values, kde = False)
    g.set(xlabel='Reflectance', ylabel='Counts', title = title)

    if save_in:
        print(f'Saving plot as {save_in} ...')
        plt.tight_layout()
        plt.savefig(save_in)
        
def calculateRange(allvalues, feature):                    
    # Calculate range of population values:
    # We need to know the distribution per feature to get fixed range for histogram:

    perc = [0,1,2,5,10,20,25,40,50,60,80,90,95,98,99,100]
    
    # removing zeros:
    
    allstripped = np.concatenate(allvalues)
    allstripped2 = allstripped[allstripped > 0]
    
    # Pickle in dict:
    stats = {'feature': feature, 'min': allstripped2.min(), 'max': allstripped2.max(),
             'n_pixels': len(allstripped), 
             'n_nonzero_pixels': len(allstripped2),
             'percentiles': perc,
             'intensities': np.percentile(allstripped2, perc)}
    
    return stats

def save_intensities(filename, arrayvalues):
    with open(filename, 'wb+') as outputfile:
        pickle.dump(arrayvalues, outputfile)

def load_intensities(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data
        
def plotPopulations(year: str, output: str, features): 
    
    filename5 = os.path.join(output, 'S2Intensities_' + year + '.png')        
    fig, axs = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)

    if 'NDBI' in features:
        #print('Distibution of NDBI population:')
        filename4 = os.path.join(output, 'populationNDBIIntensities_' + year + '.pkl')
        population = load_intensities(filename4)
        otsikko = 'Distribution of Feature NDBI'
        #plot_histogram(population, otsikko, save_in=os.path.join(output, 'population_distribution_NDBI.png'))
        sns.histplot(population, stat='density', ax=axs[0, 0], color=colors[4]).set_title(otsikko)

    if 'NDMI' in features:
        #print('Distibution of NDMI population:')
        filename4 = os.path.join(output, 'populationNDMIIntensities_' + year + '.pkl')
        population = load_intensities(filename4)
        otsikko = 'Distribution of Feature NDMI'
        #plot_histogram(population, otsikko, save_in=os.path.join(output, 'population_distribution_NDMI.png'))
        sns.histplot(population, stat='density', ax=axs[0, 1], color=colors[7]).set_title(otsikko)

    if 'NDTI' in features:
        #print('Distibution of NDTI population:')
        filename4 = os.path.join(output, 'populationNDTIIntensities_' + year + '.pkl')
        population = load_intensities(filename4)
        otsikko = 'Distribution of Feature NDTI'
        #plot_histogram(population, otsikko, save_in=os.path.join(output, 'population_distribution_NDTI.png'))
        sns.histplot(population, stat='density', ax=axs[1, 0], color=colors[6]).set_title(otsikko)
        
    if 'NDVI' in features:
        #print('Distibution of NDVI population:')
        filename4 = os.path.join(output, 'populationNDVIIntensities_' + year + '.pkl')
        population = load_intensities(filename4)
        otsikko = 'Distribution of Feature NDVI'
        #plot_histogram(population, otsikko, save_in=os.path.join(output, 'population_distribution_NDVI.png'))
        sns.histplot(population, stat='density', ax=axs[1, 1], color=colors[5]).set_title(otsikko)
        
        fig.suptitle('Distibutions of populations by S2 index', fontsize=16)
        print('Saving ' + os.path.join(output, filename5))
        plt.savefig(os.path.join(output, filename5))
        plt.close()            
        
def readImages(year: str, shpfile: str, output: str, savePopulations=False):

    # FILES NEEDED in PUHTI:

    if year == '2022': # year 2022 not in data/geo/
        fpNDVI = '/scratch/project_2002224/S2mosaics/pta_sjp_s2ind_ndvi_20220401_20220430.tif'
        fpNDTI = '/scratch/project_2002224/S2mosaics/pta_sjp_s2ind_ndti_20220401_20220430.tif'
        fpNDMI = '/scratch/project_2002224/S2mosaics/pta_sjp_s2ind_ndmi_20220401_20220430.tif'
        fpNDBI = '/scratch/project_2002224/S2mosaics/pta_sjp_s2ind_ndbi_20220401_20220430.tif'
    elif year == '2023': # year 2023 not in data/geo/
        fpNDVI = '/scratch/project_2002224/S2mosaics/pta_sjp_s2ind_ndvi_20230401_20230430.tif'
        fpNDTI = '/scratch/project_2002224/S2mosaics/pta_sjp_s2ind_ndti_20230401_20230430.tif'
        fpNDMI = '/scratch/project_2002224/S2mosaics/pta_sjp_s2ind_ndmi_20230401_20230430.tif'
        fpNDBI = '/scratch/project_2002224/S2mosaics/pta_sjp_s2ind_ndbi_20230401_20230430.tif'
    else:
        fpNDVI = '/appl/data/geo/sentinel/s2/pta_sjp_s2ind_ndvi_' + year + '0401_' + year + '0430.tif'
        fpNDTI = '/appl/data/geo/sentinel/s2/pta_sjp_s2ind_ndti_' + year + '0401_' + year + '0430.tif'
        fpNDMI = '/appl/data/geo/sentinel/s2/pta_sjp_s2ind_ndmi_' + year + '0401_' + year + '0430.tif'
        fpNDBI = '/appl/data/geo/sentinel/s2/pta_sjp_s2ind_ndbi_' + year + '0401_' + year + '0430.tif'


    myarrays = []
    populationndbi = []
    populationndmi = []
    populationndti = []
    populationndvi = []

    partialpopulationndbi = []
    partialpopulationndmi = []
    partialpopulationndti = []
    partialpopulationndvi = []
            
     
    try:
        if not os.path.exists(shpfile):
            raise Exception('Missing Shapefile with parcel polygons.')

        parcelpath = shpfile  
            
        print(f'Parcel path is {parcelpath}.')
        
    except Exception as e:
        print('\n\nUnable to read shapefile for parcels.')
        #parser.print_help()
        raise e
       

    for filename in [fpNDVI, fpNDTI, fpNDMI, fpNDBI]:
        filename_parts = filename.replace(".tif", "").split('_')

        if year in ['2022', '2023']:
            startdate = "".join(filename_parts[5:6])
            date = "".join(list(startdate)[4:])
            feature = "".join(filename_parts[4:5])
            #print(filename_parts, startdate, date, feature)
            #if year == int(vuosi) and feature in ['ndmi', 'ndti', 'ndvi']:
            print('\nProcessing Sentinel file: ' + filename.split('/')[-1])
            print('Feature: ' + "".join(filename_parts[4:5]))
            print('Mosaic year: ' + "".join(list(startdate)[:4]))
            print('Mosaic starting date: ' + "".join(filename_parts[5:6]))
        else:
            startdate = "".join(filename_parts[4:5])
            date = "".join(list(startdate)[4:])
            feature = "".join(filename_parts[3:4])
            #print(filename_parts, startdate, date, feature)
            #if year == int(vuosi) and feature in ['ndmi', 'ndti', 'ndvi']:
            print('\nProcessing Sentinel file: ' + filename.split('/')[-1])
            print('Feature: ' + "".join(filename_parts[3:4]))
            print('Mosaic year: ' + "".join(list(startdate)[:4]))
            print('Mosaic starting date: ' + "".join(filename_parts[4:5]))

        # Read all pixels within a parcel:
        parcels = zonal_stats(shpfile, filename, stats=['count', 'median'], geojson_out=True, all_touched = False, 
                              raster_out = True, nodata=-999)
        print("Length : %d" % len (parcels))

        for x in parcels:
            myarray = x['properties']['mini_raster_array'].compressed()
            if not myarray.size:
                print(f"When reading raster, got empty array for a parcel {x['properties']['parcelID']}.")
                continue
            myid = [x['properties']['parcelID'], x['properties']['target'],
                    x['properties']['count'], x['properties']['median'], startdate, date, feature]
            arr = myarray.tolist()
            myid.extend(arr)
            arr = myid

            if feature == "ndbi":
                populationndbi.append(myarray.tolist())
                partialpopulationndbi.append(myarray.tolist())
            if feature == "ndmi":
                populationndmi.append(myarray.tolist())
                partialpopulationndmi.append(myarray.tolist())
            if feature == "ndti":
                populationndti.append(myarray.tolist())
                partialpopulationndti.append(myarray.tolist())
            if feature == "ndvi":
                populationndvi.append(myarray.tolist())
                partialpopulationndvi.append(myarray.tolist())

            myarrays.append(arr)


        if feature == "ndbi":
            filename4 = os.path.join(output, 'populationNDBIIntensities_' + year + '.pkl')
            save_intensities(filename4, np.histogram(np.concatenate(partialpopulationndbi), bins = 32, density=False) )
        if feature == "ndmi":
            filename4 = os.path.join(output, 'populationNDMIIntensities_' + year + '.pkl')
            save_intensities(filename4, np.histogram(np.concatenate(partialpopulationndmi), bins = 32, density=False) )
        if feature == "ndti":
            filename4 = os.path.join(output, 'populationNDTIIntensities_' + year + '.pkl')
            save_intensities(filename4, np.histogram(np.concatenate(partialpopulationndti), bins = 32, density=False) )
        if feature == "ndvi":
            filename4 = os.path.join(output, 'populationNDVIIntensities_' + year + '.pkl')
            save_intensities(filename4, np.histogram(np.concatenate(partialpopulationndvi), bins = 32, density=False) )

    # Save all pixel values:    
    filename2 = os.path.join(output, 'fullArrayIntensities' + year + 'S2.pkl')
    print('Saving all pixels values into ' + filename2 + '...')
    save_intensities(filename2, myarrays)
    
    if savePopulations:

        filename4 = os.path.join(output, 'populationNDBIIntensities_' + year + '.pkl')
        save_intensities(filename4, np.concatenate(populationndbi))
        filename4 = os.path.join(output, 'populationNDMIIntensities_' + year + '.pkl')
        save_intensities(filename4, np.concatenate(populationndmi))
        filename4 = os.path.join(output, 'populationNDTIIntensities_' + year + '.pkl')
        save_intensities(filename4, np.concatenate(populationndti))
        filename4 = os.path.join(output, 'populationNDVIIntensities_' + year + '.pkl')
        save_intensities(filename4, np.concatenate(populationndvi))
    



    # calculatePopulations(outputdir: str, plotPopulations=False, show=False):
    
    populationStats = [calculateRange(populationndbi, 'ndbi'),
                       calculateRange(populationndmi, 'ndmi'),
                       calculateRange(populationndti, 'ndti'),
                       calculateRange(populationndvi, 'ndvi')]

    filename3 = os.path.join(output, 'populationStatsS2.pkl')
    print('Saving population statistics into ' + filename3 + '...')
    save_intensities(filename3, populationStats)

def decideBinSeq(outputdirsetti):
    
    print('We choose values for range:')
    # 25.9.2023 modified:
    ndbirange = [26, 110]
    ndmirange = [70, 140]
    ndtirange = [107, 133]
    ndvirange = [105, 152]
    
    return ndbirange, ndmirange, ndtirange, ndvirange

def makeHisto(outputdir, bins, vuosi):
    print('\nDecide bin sequences for range...')
    ndbirange, ndmirange, ndtirange, ndvirange = decideBinSeq(outputdir)
      
    histlist = []
    nrbins = bins

    filename = os.path.join(outputdir, 'fullArrayIntensities' + vuosi + 'S2.pkl')

    print('\nCalculate histograms...')

    with open(filename, "rb") as f:
        tmp = pickle.load(f)

        for elem in tmp:
            myid = elem[0:7]
            line = elem[7:]
            inx = elem[6]
            
            if not line:
                print("Line is empty")
                continue

            if inx == 'ndbi':
                maximum = ndbirange[1]
                minimum = ndbirange[0]
                bin_seq = np.linspace(minimum,maximum,nrbins+1)

            elif inx == 'ndmi':
                maximum = ndmirange[1]
                minimum = ndmirange[0]
                bin_seq = np.linspace(minimum,maximum,nrbins+1)

            elif inx == 'ndti':
                maximum = ndtirange[1]
                minimum = ndtirange[0]
                bin_seq = np.linspace(minimum,maximum,nrbins+1)

            elif inx == 'ndvi':
                maximum = ndvirange[1]
                minimum = ndvirange[0]
                bin_seq = np.linspace(minimum,maximum,nrbins+1)
                
            else:
                print(f'Index {inx} not found!')
                print(f' {myid, line, elem}')
                break

            if min(line) >= maximum:
                hist = [float(0)]*(nrbins-1); hist.append(float(1)) 
            elif max(line) <= minimum:
                hist = [1]; hist.extend([0]*(nrbins-1))
            else:
                #density, _ = np.histogram(line, bin_seq, density=False)
                #hist = density / float(density.sum())
                hist, _ = np.histogram(line, bin_seq, density=False)

            myid.extend(hist)
            hist2 = myid
            #print(hist2)
            histlist.append(hist2)

    print('Saving histograms into histograms' + vuosi + '.pkl...')
    save_intensities(os.path.join(outputdir, 'histograms' + vuosi + 'S2.pkl'), histlist)        



# HERE STARTS MAIN:

def main(args):

    try:
        if not args.year or not args.outputpath:
            raise Exception('Missing year or output directory arguments. Try --help .')

        print(f'\n01-makeS2IndexHisto.py')
        print(f'\nReading S2 image mosaics for the year {args.year}.')
   
        vuosi = args.year
        nrbins = 32
        
        t0 = time.time()

    
        out_dir_path = Path(os.path.join(args.outputpath, args.year + args.startdate))
        print(out_dir_path)
        out_dir_path.mkdir(parents=True, exist_ok=True)            

        
        if args.extractIntensities:
            readImages(vuosi, args.parcels, out_dir_path, savePopulations=True)
            
        if args.plotPopulations:
            plotPopulations(vuosi, out_dir_path, args.alist)
            
        if args.make_histograms:
            makeHisto(out_dir_path, nrbins, vuosi)

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
    parser.add_argument('-i', '--parcels',
                        type = str,
                        help = 'Parcels in Shapefile.',
                        default='.')
    parser.add_argument('-f', '--features', action='store', dest='alist',
                       type=str, nargs='*', default=['NDVI', 'NDTI', 'NDMI', 'NDBI'],
                       help="For plotting only, optionally e.g. -f NDVI NDTI, default all")
    parser.add_argument('-o', '--outputpath',
                        type=str,
                        help='Directory for output files.',
                        default='.')
    parser.add_argument('-p', '--extractIntensities',
                        help='Extract all pixel values from parcels.',
                        default=False,
                        action='store_true')
    parser.add_argument('-n', '--plotPopulations',
                        help="Draw histograms of each features's pixel profile.",
                        default=False,
                        action='store_true')  
    parser.add_argument('-m', '--make_histograms',
                        help="Extract histograms of (multi)parcel's pixel profile.",
                        default=False,
                        action='store_true')   
    parser.add_argument('--debug',
                        help='Verbose output for debugging.',
                        action='store_true')

    args = parser.parse_args()
    main(args)
