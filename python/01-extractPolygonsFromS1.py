#!/usr/bin/env python
# coding: utf-8

"""
2022-10-04 MY
2023-01-11 Modified for 4 classes

Pixel values into fullArray files. Plots fo distributions.

env: myGIS or module load geoconda
From Sentinel-1 mosaics to full array data with rasterstats. Features 'VV', 'VH'


INPUTS:
i: input shapefile of parcels
y: year
o: outputdir
d: start date


OUTPUT:

RUN:

python ../python/01-extractPolygonsFromS1.py -y 2023 --startdate 0411 \
     --shppolku /Users/myliheik/Documents/myVEGCOVER/vegcover2023/shpfiles/insitu/extendedReferences2023-AOI-sampled-buffered.shp \
    -o /Users/myliheik/Documents/myVEGCOVER/vegcover2023/S1/results-insitu/ \
    --extractIntensities --savePopulations --plotPopulations
    
    
NEXT: 
Run 02-makeS1Histo.py

"""


import argparse
import matplotlib.pyplot as plt
import numpy as np
import os.path
import os
import pandas as pd
import geopandas as gpd
import shapely
import fiona
from pathlib import Path
import pickle

from rasterstats import zonal_stats
import seaborn as sns
import textwrap

import time

sns.set_style('darkgrid')

colors = ["black", "goldenrod" , "saddlebrown", "gold", "skyblue", "yellowgreen", "darkseagreen", "darkgoldenrod"]

def save_intensities(filename, arrayvalues):
    with open(filename, 'wb+') as outputfile:
        pickle.dump(arrayvalues, outputfile)
        
def load_intensities(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data
        
def calculateRange(allvalues, feature):                    
    # Calculate range of population values:
    # We need to know the distribution per feature to get fixed range for histogram:

    perc = [0,1,2,5,10,20,25,40,50,60,80,90,95,98,99,100]
       
    allstripped = np.concatenate(allvalues)
    if feature in ['vv', 'vh']:
        allstripped2 = allstripped[allstripped < 0] 
        #print(allstripped)
    else:
        allstripped2 = allstripped[(allstripped > -4) & (allstripped < 18)]
        
    # Pickle in dict:
    if len(allstripped2) > 0:
        stats = {'feature': feature, 'min': allstripped2.min(), 'max': allstripped2.max(),
                 'n_pixels': len(allstripped), 
                 'n_nonzero_pixels': len(allstripped2),
                 'percentiles': perc,
                 'intensities': np.percentile(allstripped2, perc)}
    else:
        stats = {'feature': feature, 'min': None, 'max': None,
                 'n_pixels': len(allstripped), 
                 'n_nonzero_pixels': len(allstripped2),
                 'percentiles': perc,
                 'intensities': None}    
    
    return stats
 

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
    else:
        print('Start date did not match any of our time ranges')
        date = None 
    
    return date
    
def read_shapes_within_bounds(shapefile_name: str, bounding_box: shapely.geometry.polygon.Polygon) -> ([dict], [str], [str]):
    parcelIDsInSubset = []
    with fiona.open(shapefile_name, 'r') as shapefile:
        for feature in shapefile:
            shape = shapely.geometry.shape(feature['geometry'])
            if bounding_box.contains(shape):
                parcelIDsInSubset.append(feature['properties']['parcelID'])
    return parcelIDsInSubset

def plot_histogram(values, title:str, save_in=None):
    g = sns.displot(values, kde = False)
    g.set(xlabel='Log(dB)', ylabel='Counts', title = title)

    if save_in:
        print(f'Saving plot as {save_in}')
        plt.tight_layout()
        plt.savefig(save_in)

def plotPopulationsFunc(year: str, output: str, features):        
        
    if 'VH' in features:
        print('Distibution of VH population:')
        filename4 = os.path.join(output, 'populationVHIntensities_' + year + '.pkl')
        populationvh = load_intensities(filename4)
        otsikko = 'Distribution of Feature VH'
        plot_histogram(populationvh, otsikko, save_in=os.path.join(output, 'population_distribution_VH.png'))
        
        
        filename5 = os.path.join(output, 'populationVHIntensities_perClass_' + year + '.png')        
        fig, axs = plt.subplots(2, 3, figsize=(9, 7), sharex=True, sharey=True)
        
        ###
        filename4 = os.path.join(output, 'ploughVHIntensities_' + year + '.pkl')
        if os.path.exists(filename4):
            ploughvh = load_intensities(filename4)
            sns.histplot(ploughvh[ploughvh < 0], stat='density', ax=axs[0, 0], color=colors[0]).set_title("Plough")

        filename4 = os.path.join(output, 'conservationVHIntensities_' + year + '.pkl')
        if os.path.exists(filename4):
            conservationvh = load_intensities(filename4)
            sns.histplot(conservationvh[conservationvh < 0], stat='density', ax=axs[0, 1], color=colors[7]).set_title("Conservation tillage")

        filename4 = os.path.join(output, 'autumnVHIntensities_' + year + '.pkl')
        if os.path.exists(filename4):
            autumnvh = load_intensities(filename4)
            sns.histplot(autumnvh[autumnvh < 0], stat='density', ax=axs[0, 2], color=colors[5]).set_title("Autumn crop")

        filename4 = os.path.join(output, 'grassVHIntensities_' + year + '.pkl')
        if os.path.exists(filename4):
            grassvh = load_intensities(filename4)
            sns.histplot(grassvh[grassvh < 0], stat='density', ax=axs[1, 0], color=colors[6]).set_title("Grass")

        filename4 = os.path.join(output, 'stubbleVHIntensities_' + year + '.pkl')
        if os.path.exists(filename4):
            stubblevh = load_intensities(filename4)
            sns.histplot(stubblevh[stubblevh < 0], stat='density', ax=axs[1, 1], color=colors[1]).set_title("Stubble")

        filename4 = os.path.join(output, 'stubbleCerealVHIntensities_' + year + '.pkl')
        if os.path.exists(filename4):
            stubbleCerealvh = load_intensities(filename4)
            sns.histplot(stubbleCerealvh[stubbleCerealvh < 0], stat='density', ax=axs[1, 1], color=colors[2]).set_title("Stubble after cereal crop")

        filename4 = os.path.join(output, 'stubbleCompVHIntensities_' + year + '.pkl')
        if os.path.exists(filename4):
            stubbleCompvh = load_intensities(filename4)
            sns.histplot(stubbleCompvh[stubbleCompvh < 0], stat='density', ax=axs[1, 2], color=colors[3]).set_title("Stubble with companion crop")

        fig.suptitle('Distibutions of VH population per class', fontsize=16)
        fig.supxlabel('Log(dB)')
        print('Saving ' + os.path.join(output, filename5))
        plt.savefig(os.path.join(output, filename5))
        plt.close()            
        
    if 'VV' in features:
        print('Distibution of VV population:')
        filename4 = os.path.join(output, 'populationVVIntensities_' + year + '.pkl')
        populationvv = load_intensities(filename4)
        otsikko = 'Distribution of Feature VV'
        plot_histogram(populationvv, otsikko, save_in=os.path.join(output, 'population_distribution_VV.png'))        
        
        filename5 = os.path.join(output, 'populationVVIntensities_perClass_' + year + '.png')        
        fig, axs = plt.subplots(2, 3, figsize=(9, 7), sharex=True, sharey=True)
        
        ###
        filename4 = os.path.join(output, 'ploughVVIntensities_' + year + '.pkl')
        if os.path.exists(filename4):
            ploughvv = load_intensities(filename4)
            sns.histplot(ploughvv[ploughvv < 0], stat='density', ax=axs[0, 0], color=colors[0]).set_title("Plough")

        filename4 = os.path.join(output, 'conservationVVIntensities_' + year + '.pkl')
        if os.path.exists(filename4):
            conservationvv = load_intensities(filename4)
            sns.histplot(conservationvv[conservationvv < 0], stat='density', ax=axs[0, 1], color=colors[7]).set_title("Conservation tillage")

        filename4 = os.path.join(output, 'autumnVVIntensities_' + year + '.pkl')
        if os.path.exists(filename4):
            autumnvv = load_intensities(filename4)
            sns.histplot(autumnvv[autumnvv < 0], stat='density', ax=axs[0, 2], color=colors[5]).set_title("Autumn crop")

        filename4 = os.path.join(output, 'grassVVIntensities_' + year + '.pkl')
        if os.path.exists(filename4):
            grassvv = load_intensities(filename4)
            sns.histplot(grassvv[grassvv < 0], stat='density', ax=axs[1, 0], color=colors[6]).set_title("Grass")

        filename4 = os.path.join(output, 'stubbleVVIntensities_' + year + '.pkl')
        if os.path.exists(filename4):
            stubblevv = load_intensities(filename4)
            sns.histplot(stubblevv[stubblevv < 0], stat='density', ax=axs[1, 1], color=colors[1]).set_title("Stubble")

        filename4 = os.path.join(output, 'stubbleCerealVVIntensities_' + year + '.pkl')
        if os.path.exists(filename4):
            stubbleCerealvv = load_intensities(filename4)
            sns.histplot(stubbleCerealvv[stubbleCerealvv < 0], stat='density', ax=axs[1, 1], color=colors[2]).set_title("Stubble after cereal crop")

        filename4 = os.path.join(output, 'stubbleCompVVIntensities_' + year + '.pkl')
        if os.path.exists(filename4):
            stubbleCompvv = load_intensities(filename4)
            sns.histplot(stubbleCompvv[stubbleCompvv < 0], stat='density', ax=axs[1, 2], color=colors[3]).set_title("Stubble with companion crop")

        fig.suptitle('Distibutions of VV population per class', fontsize=16)
        fig.supxlabel('Log(dB)')
        print('Saving ' + filename5)
        plt.savefig(os.path.join(output, filename5))
        plt.close()    
        
    if features == ['VV', 'VH']:
        f, ax = plt.subplots(1, 1)
        sns.histplot(populationvv, kde = False, label = 'VV', ax = ax)
        sns.histplot(populationvh, kde = False, label = 'VH', ax = ax)
        ax.set(xlabel='Log(dB)', ylabel='Counts', title = f'VV and VH distributions in {year}')
        ax.legend()
        print('Saving ' + os.path.join(output, 'population_distribution.png'))
        plt.tight_layout()  
        plt.savefig(os.path.join(output, 'population_distribution.png'))
        plt.close()    
        
def readImages(year: str, startdate: str, date: str, shppolku: str, aoi_shapefile: str, output: str, features, zoneID, savePopulations=False, plotPopulations=False):
    
    # s1 files in Puhti:
    #fpVV = '/appl/data/geo/sentinel/s1/s1m_grd_' + date + '_mean_VV_R20m.tif'
    #fpVH = '/appl/data/geo/sentinel/s1/s1m_grd_' + date + '_mean_VH_R20m.tif'

    
    fpVV = '/Users/myliheik/Documents/myVEGCOVER/S1/s1m_grd_' + date + '_mean_VV_R20m.tif'
    fpVH = '/Users/myliheik/Documents/myVEGCOVER/S1/s1m_grd_' + date + '_mean_VH_R20m.tif'

    #############################
    # If spring zone or other AOI is given, we mask the subset with zone boundaries (bbox):    
    if aoi_shapefile:
        if zoneID:
            #############################
            # Read in zone file:   
            gdf = gpd.read_file(aoi_shapefile)
            newgdf = gdf[gdf['Jako3Tunnu'].isin(zoneID)]
            # make bbox:
            bbox = newgdf.total_bounds
            polygon = shapely.geometry.box(*bbox, ccw=True)
        else:
            gdf = gpd.read_file(aoi_shapefile)
            bbox = gdf.total_bounds
            polygon = shapely.geometry.box(*bbox, ccw=True)

    myarrays = []
    populationvv = []
    populationvh = []
    populationvvvh = []

    if zoneID:
        print(f'\nProcessing zone {zoneID}...')

    shpfile = os.path.join(shppolku) 

    try:
        if not os.path.exists(shpfile):
            raise Exception('Missing Shapefile with parcel polygons.')

        if aoi_shapefile:    
            #############################
            # Mask 
            print(f'\nFiltering a subset polygons within zone bounds of {bbox}...')
            properties = read_shapes_within_bounds(shpfile, polygon)

            if len(properties) == 0:
                #raise Exception('Found no polygons within zone bounds. Do zone and .shp projections match?')
                print('Found no polygons within zone bounds. Proceeding to next data set')



            print(f'Found {len(properties)} polygons within zone bounds.  \n')#Extracting pixels...')

            gdf = gpd.read_file(shpfile)
            gdf2 = gdf[gdf['parcelID'].isin(properties)]
            # We need to write this to temp file for zonal_stats:
            gdf2.to_file(os.path.join(shppolku[:-4] + '_AOI.shp') )
            parcelpath = os.path.join(shppolku[:-4] + '_AOI.shp')          
        else:
            parcelpath = shpfile  
            
        print(f'Parcel path is {parcelpath}.')
        
    except Exception as e:
        print('\n\nUnable to read input or write out results. Check prerequisites and see exception output below.')
        #parser.print_help()
        raise e
    
    ##################################################
    # read features:
    
    if 'VV' in features:

        ploughvv = []
        conservationvv = []
        autumnvv = []
        grassvv = []
        stubblevv = []
        stubbleCerealvv = []
        stubbleCompvv = []

        filename = fpVV
        filename_parts = filename.replace(".tif", "").split('_')

        date0 = filename_parts[2]
        featureset = filename_parts[4]
        feature = "vv"

        print('\nProcessing Sentinel file: ' + filename.split('/')[-1])
        print('Feature: ' + featureset)
        print('Mosaic year: ' + "".join(list(date0)[:4]))
        print('Mosaic starting date: ' + date0[4:8])

        raster_file = filename

        # Read all pixels within a parcel: 

        parcelsVV = zonal_stats(parcelpath, filename, stats=['count', 'median'], geojson_out=True, all_touched = False, 
                              raster_out = True, nodata=-32767)
        print("Length : %d" % len (parcelsVV))

        for x in parcelsVV:
            myarray = x['properties']['mini_raster_array'].compressed()
            if not myarray.size:
                print(f"When reading raster, got empty array for a parcel {x['properties']['parcelID']}.")
                continue
            #print(x['properties'])
            
            myid = [x['properties']['parcelID'], x['properties']['target'],
                    x['properties']['count'], x['properties']['median'], startdate, date, feature]
            #myid = [x['parcelID'], x['target'], x['split'], 
            #        x['PINTAALA'], x['properties']['mean'], x['properties']['median'], startdate, feature]
            #print(myid)
            arr = myarray.tolist()
            myid.extend(arr)
            arr = myid

            myarrays.append(arr)

            if savePopulations:
                populationvv.append(myarray.tolist())

                if x['properties']['target'] == 0:
                    ploughvv.append(myarray.tolist())
                elif x['properties']['target'] == 1:
                    conservationvv.append(myarray.tolist())
                elif x['properties']['target'] == 2:
                    autumnvv.append(myarray.tolist())
                elif x['properties']['target'] == 3:
                    grassvv.append(myarray.tolist())
                elif x['properties']['target'] == 4:
                    stubblevv.append(myarray.tolist())                
                elif x['properties']['target'] == 41:
                    stubbleCerealvv.append(myarray.tolist())
                elif x['properties']['target'] == 5:
                    stubbleCompvv.append(myarray.tolist())
                else:
                    print('Target value did not match any of these: 0, 1, 2, 3, 4, 41, 5')

        if savePopulations: 
            if ploughvv:
                filename4 = os.path.join(output, 'ploughVVIntensities_' + year + '.pkl')
                save_intensities(filename4, np.concatenate(ploughvv))
            if conservationvv:
                filename4 = os.path.join(output, 'conservationVVIntensities_' + year + '.pkl')
                save_intensities(filename4, np.concatenate(conservationvv))
            if autumnvv:
                filename4 = os.path.join(output, 'autumnVVIntensities_' + year + '.pkl')
                save_intensities(filename4, np.concatenate(autumnvv))
            if grassvv:
                filename4 = os.path.join(output, 'grassVVIntensities_' + year + '.pkl')
                save_intensities(filename4, np.concatenate(grassvv))
            if stubblevv:
                filename4 = os.path.join(output, 'stubbleVVIntensities_' + year + '.pkl')
                save_intensities(filename4, np.concatenate(stubblevv))
            if stubbleCerealvv:
                filename4 = os.path.join(output, 'stubbleCerealVVIntensities_' + year + '.pkl')
                save_intensities(filename4, np.concatenate(stubbleCerealvv))        
            if stubbleCompvv:
                filename4 = os.path.join(output, 'stubbleCompVVIntensities_' + year + '.pkl')
                save_intensities(filename4, np.concatenate(stubbleCompvv))    

    ##################################################
    # VH
    if 'VH' in features:

        ploughvh = []
        conservationvh = []
        autumnvh = []
        grassvh = []
        stubblevh = []
        stubbleCerealvh = []
        stubbleCompvh = []

        filename = fpVH
        filename_parts = filename.replace(".tif", "").split('_')

        date0 = filename_parts[2]
        featureset = filename_parts[4]
        #date = "".join(list(date0)[4:8]) # alkupvm
        feature = "vh"

        print('\nProcessing Sentinel file: ' + filename.split('/')[-1])
        print('Feature: ' + featureset)
        print('Mosaic year: ' + date0[:4])
        print('Mosaic starting date: ' + filename_parts[2][4:8])

        # Read data
        # Worker function needs access to data, so these need to be global variables.

        raster_file = filename


        # Read all pixels within a parcel: 
        parcelsVH = zonal_stats(parcelpath, filename, stats=['count', 'median'], geojson_out=True, all_touched = False, 
                              raster_out = True, nodata=-32767)
        print("Length : %d" % len (parcelsVH))


        for x in parcelsVH:
            myarray = x['properties']['mini_raster_array'].compressed()
            if not myarray.size:
                print(f"When reading raster, got empty array for a parcel {x['properties']['parcelID']}.")

            myid = [x['properties']['parcelID'], x['properties']['target'],
                    x['properties']['count'], x['properties']['median'], startdate, date, feature]

            arr = myarray.tolist()
            myid.extend(arr)
            arr = myid

            myarrays.append(arr)  

            if savePopulations:
                populationvh.append(myarray.tolist())

                if x['properties']['target'] == 0:
                    ploughvh.append(myarray.tolist())
                elif x['properties']['target'] == 1:
                    conservationvh.append(myarray.tolist())
                elif x['properties']['target'] == 2:
                    autumnvh.append(myarray.tolist())
                elif x['properties']['target'] == 3:
                    grassvh.append(myarray.tolist())
                elif x['properties']['target'] == 4:
                    stubblevh.append(myarray.tolist())                
                elif x['properties']['target'] == 41:
                    stubbleCerealvh.append(myarray.tolist())
                elif x['properties']['target'] == 5:
                    stubbleCompvh.append(myarray.tolist())
                else:
                    print('Target value did not match any of these: 0, 1, 2, 3, 4, 41, 5')

        if savePopulations: 
            if ploughvh:
                filename4 = os.path.join(output, 'ploughVHIntensities_' + year + '.pkl')
                save_intensities(filename4, np.concatenate(ploughvh))
            if conservationvh:
                filename4 = os.path.join(output, 'conservationVHIntensities_' + year + '.pkl')
                save_intensities(filename4, np.concatenate(conservationvh))
            if autumnvh:
                filename4 = os.path.join(output, 'autumnVHIntensities_' + year + '.pkl')
                save_intensities(filename4, np.concatenate(autumnvh))
            if grassvh:
                filename4 = os.path.join(output, 'grassVHIntensities_' + year + '.pkl')
                save_intensities(filename4, np.concatenate(grassvh))
            if stubblevh:
                filename4 = os.path.join(output, 'stubbleVHIntensities_' + year + '.pkl')
                save_intensities(filename4, np.concatenate(stubblevh))
            if stubbleCerealvh:
                filename4 = os.path.join(output, 'stubbleCerealVHIntensities_' + year + '.pkl')
                save_intensities(filename4, np.concatenate(stubbleCerealvh))        
            if stubbleCompvh:
                filename4 = os.path.join(output, 'stubbleCompVHIntensities_' + year + '.pkl')
                save_intensities(filename4, np.concatenate(stubbleCompvh))



    ###################################################
    # Save all pixel values:     
    filename2 = os.path.join(output, 'fullArrayIntensities' + year + 'S1-' + '_'.join(features) + '.pkl')    
    print('Saving all pixels values into ' + filename2)
    save_intensities(filename2, myarrays)
    print('yes')
    print(len(np.concatenate(populationvh)), len(np.concatenate(populationvv)))
    
    if savePopulations:
        if 'VV' in features:
            filename4 = os.path.join(output, 'populationVVIntensities_' + year + '.pkl')
            save_intensities(filename4, np.concatenate(populationvv))        
        if 'VH' in features:
            filename4 = os.path.join(output, 'populationVHIntensities_' + year + '.pkl')
            save_intensities(filename4, np.concatenate(populationvh))            
        
    #def calculatePopulations(outputdir: str, plotPopulations=False, show=False):
    if savePopulations:
        if features == ['VV', 'VH']:      
            populationStats = [calculateRange(populationvv, 'vv'),
                               calculateRange(populationvh, 'vh')]

            filename3 = os.path.join(output, 'populationStatsS1.pkl')

            print('Saving population statistics into ' + filename3)
            save_intensities(filename3, populationStats)
        else:
            print('Population statistics not saved. You need to run both 2 features to calculate statistics.')
    
    #if plotPopulations:
    #    plotPopulationsFunc(year, output, features)
    
# HERE STARTS MAIN:

def main(args):

    try:
        if not args.year or not args.outputpath:
            raise Exception('Missing year or output directory arguments. Try --help .')

        print(f'\n01-extractPolygonsFromS1.py')
        print(f'\nReading S1 image mosaics for the year {args.year}.')
   
        t0 = time.time()

        nrbins = 16
    
        out_dir_path = Path(os.path.expanduser(args.outputpath + args.year + args.startdate))
        out_dir_path.mkdir(parents=True, exist_ok=True)
        
        if args.extractIntensities:
            date = parseDate(args.startdate, args.year)
            readImages(args.year, args.startdate, date, args.shppolku, args.aoi_shapefile, out_dir_path, args.alist, args.zonelist, savePopulations=args.savePopulations, plotPopulations=args.plotPopulations)
            
        if args.plotPopulations:
            plotPopulationsFunc(args.year, out_dir_path, args.alist)
            
            
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
    
    parser.add_argument('-z', '--zones', action='store', dest='zonelist',
                       type=str, nargs='*', default=None,
                       help="Optionally e.g. -z zone1 zone2, default all")


    parser.add_argument('-o', '--outputpath',
                        type=str,
                        help='Directory for output files.',
                        default='.')
    parser.add_argument('-s', '--shppolku',
                        type=str,
                        help='Filepath polygon shp file.',
                        default='.')
    parser.add_argument('-a', '--aoi_shapefile',
                        type=str,
                        help='Path to shp file where is bbox for AOI.',
                        default=None)

    parser.add_argument('-p', '--extractIntensities',
                        help='Extract all pixel values from parcels.',
                        default=False,
                        action='store_true')
    parser.add_argument('-q', '--savePopulations',
                        help="Save of each features's pixel profile. Needed if you want to draw histograms later.",
                        default=False,
                        action='store_true') 
    parser.add_argument('-n', '--plotPopulations',
                        help="Draw histograms of each features's pixel profile.",
                        default=False,
                        action='store_true')   
    parser.add_argument('--debug',
                        help='Verbose output for debugging.',
                        action='store_true')

    args = parser.parse_args()
    main(args)

    
