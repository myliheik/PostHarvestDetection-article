#!/usr/bin/env python
# coding: utf-8

"""
2022-10-04 MY
2023-12-30 Modified InSAR

Pixel values into fullArray files. Plots fo distributions.

nodata = 0 

env: myGIS or module load geoconda
From 12-days InSAR to full array data with rasterstats. Feature 'COH12'


INPUTS:
i: input shapefile of parcels
y: year
o: outputdir
d: start date


OUTPUT:

RUN:

python ../python/01-extractPolygonsFromInSAR.py -y 2023 --startdate 0408 \
     --shppolku /projappl/project_2002224/vegcover2023/shpfiles/insitu/extendedReferences2023-AOI-sampled-buffered.shp \
    -o /scratch/project_2002224/vegcover2023/InSAR/ \
    --extractIntensities --savePopulations --plotPopulations
    
    
NEXT: 
Run 02-makeInSARHisto.py

"""
import glob

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

from osgeo import gdal, osr, ogr
import rasterio

from rasterstats import zonal_stats
import seaborn as sns
import textwrap

import time

sns.set_style('darkgrid')

colors = ["black", "goldenrod" , "saddlebrown", "gold", "skyblue", "yellowgreen", "darkseagreen", "darkgoldenrod"]

def save_COH12(filename, arrayvalues):
    with open(filename, 'wb+') as outputfile:
        pickle.dump(arrayvalues, outputfile)
        
def load_intensities(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def parseDate(startdate, year):    
    if startdate == '0408':
        enddate = '0420'
        date = year + '0408-' + year + enddate        
    else:
        print('Start date did not match any of our time ranges')
        date = None 
        enddate = None
    
    return date, enddate
    
def bb_to_polygon(bb: rasterio.coords.BoundingBox) -> shapely.geometry.Polygon:
    return shapely.geometry.Polygon([(bb.left, bb.top),
                                     (bb.right, bb.top),
                                     (bb.right, bb.bottom),
                                     (bb.left, bb.bottom),
                                     (bb.left, bb.top)])


def read_shapes_within_bounds(shapefile_name: str, bounding_box: shapely.geometry.polygon.Polygon) -> ([dict], [str], [str]):
    filtered_shapes = []
    partitioning_ix = []
    target = []
    parcelIDs = []
    with fiona.open(shapefile_name, 'r') as shapefile:
        for feature in shapefile:
            shape = shapely.geometry.shape(feature['geometry'])
            if bounding_box.contains(shape):
                filtered_shapes.append(feature['geometry'])
                #print(feature['properties']['parcelID'])
                parcelIDs.append(feature['properties']['parcelID'])
                if 'split' in feature['properties']:
                    partitioning_ix.append(feature['properties']['split'])
                    target.append(feature['properties']['target'])
                else:
                    partitioning_ix.append('N/A')
                    target.append('N/A')
    return filtered_shapes, parcelIDs, partitioning_ix, target


def plot_histogram(values, title:str, save_in=None):
    g = sns.displot(values, kde = False)
    g.set(xlabel='Coherence', ylabel='Counts', title = title)

    if save_in:
        print(f'Saving plot as {save_in}')
        plt.tight_layout()
        plt.savefig(save_in)

def plotPopulationsFunc(year: str, output: str, features):        
    perc = [0,1,2,5,10,20,25,40,50,60,80,90,95,98,99,100]
    
    if 'VH' in features:
        print('Distibution of VH COH12:')
        filename4 = os.path.join(output, 'populationCOH12VHIntensities_' + year + '.pkl')
        populationvh = load_intensities(filename4)
        
        stats = {'feature': 'COH12VH', 'min': populationvh.min(), 'max': populationvh.max(),
     'n_pixels': len(populationvh), 
     'n_nonzero_pixels': len(populationvh[populationvh > 0]),
     'percentiles': perc,
     'intensities': np.percentile(populationvh, perc)}
        print(stats)
            
        otsikko = 'Distribution of Feature VH COH12'
        # Kumpi?
        #plot_histogram(populationvh, otsikko, save_in=os.path.join(output, 'population_distribution_VH.png'))
        plot_histogram(populationvh[populationvh > 0], otsikko, save_in=os.path.join(output, 'population_distribution_VH.png'))
        
        
        filename5 = os.path.join(output, 'populationCOH12VH_perClass_' + year + '.png')        
        fig, axs = plt.subplots(2, 3, figsize=(9, 7), sharex=True, sharey=True)
        
        ###
        filename4 = os.path.join(output, 'ploughVHCOH12_' + year + '.pkl')
        if os.path.exists(filename4):
            ploughvh = load_intensities(filename4)
            sns.histplot(ploughvh[ploughvh > 0], stat='density', ax=axs[0, 0], color=colors[0]).set_title("Plough")

        filename4 = os.path.join(output, 'conservationVHCOH12_' + year + '.pkl')
        if os.path.exists(filename4):
            conservationvh = load_intensities(filename4)
            sns.histplot(conservationvh[conservationvh > 0], stat='density', ax=axs[0, 1], color=colors[7]).set_title("Conservation tillage")

        filename4 = os.path.join(output, 'autumnVHCOH12_' + year + '.pkl')
        if os.path.exists(filename4):
            autumnvh = load_intensities(filename4)
            sns.histplot(autumnvh[autumnvh > 0], stat='density', ax=axs[0, 2], color=colors[5]).set_title("Autumn crop")

        filename4 = os.path.join(output, 'grassVHCOH12_' + year + '.pkl')
        if os.path.exists(filename4):
            grassvh = load_intensities(filename4)
            sns.histplot(grassvh[grassvh > 0], stat='density', ax=axs[1, 0], color=colors[6]).set_title("Grass")

        #filename4 = os.path.join(output, 'stubbleVHCOH12_' + year + '.pkl')
        #if os.path.exists(filename4):
        #    stubblevh = load_intensities(filename4)
        #    sns.histplot(stubblevh[stubblevh > 0], stat='density', ax=axs[1, 0], color=colors[1]).set_title("Stubble")

        filename4 = os.path.join(output, 'stubbleCerealVHCOH12_' + year + '.pkl')
        if os.path.exists(filename4):
            stubbleCerealvh = load_intensities(filename4)
            sns.histplot(stubbleCerealvh[stubbleCerealvh > 0], stat='density', ax=axs[1, 1], color=colors[2]).set_title("Stubble after cereal crop")

        filename4 = os.path.join(output, 'stubbleCompVHCOH12_' + year + '.pkl')
        if os.path.exists(filename4):
            stubbleCompvh = load_intensities(filename4)
            sns.histplot(stubbleCompvh[stubbleCompvh > 0], stat='density', ax=axs[1, 2], color=colors[3]).set_title("Stubble with companion crop")

        fig.suptitle('Distibutions of VH COH12 population per class', fontsize=16)
        fig.supxlabel('Coherence')
        print('Saving ' + os.path.join(output, filename5))
        plt.savefig(os.path.join(output, filename5))
        plt.close()            
        
    if 'VV' in features:
        print('Distibution of VV COH12 population:')
        filename4 = os.path.join(output, 'populationCOH12VVIntensities_' + year + '.pkl')
        populationvv = load_intensities(filename4)
        
        stats = {'feature': 'COH12VV', 'min': populationvv.min(), 'max': populationvv.max(),
         'n_pixels': len(populationvv), 
         'n_nonzero_pixels': len(populationvv[populationvv > 0]),
         'percentiles': perc,
         'intensities': np.percentile(populationvv, perc)}
        print(stats)        
        
        otsikko = 'Distribution of Feature VV COH12'
        #plot_histogram(populationvv, otsikko, save_in=os.path.join(output, 'population_distribution_VV.png'))        
        plot_histogram(populationvv[populationvv > 0], otsikko, save_in=os.path.join(output, 'population_distribution_VV.png'))
        
        filename5 = os.path.join(output, 'populationCOH12VV_perClass_' + year + '.png')        
        fig, axs = plt.subplots(2, 4, figsize=(9, 7), sharex=True, sharey=True)
        
        ###
        filename4 = os.path.join(output, 'ploughVVCOH12_' + year + '.pkl')
        if os.path.exists(filename4):
            ploughvv = load_intensities(filename4)
            sns.histplot(ploughvv[ploughvv > 0], stat='density', ax=axs[0, 0], color=colors[0]).set_title("Plough")

        filename4 = os.path.join(output, 'conservationVVCOH12_' + year + '.pkl')
        if os.path.exists(filename4):
            conservationvv = load_intensities(filename4)
            sns.histplot(conservationvv[conservationvv > 0], stat='density', ax=axs[0, 1], color=colors[7]).set_title("Conservation tillage")

        filename4 = os.path.join(output, 'autumnVVCOH12_' + year + '.pkl')
        if os.path.exists(filename4):
            autumnvv = load_intensities(filename4)
            sns.histplot(autumnvv[autumnvv > 0], stat='density', ax=axs[0, 2], color=colors[5]).set_title("Autumn crop")

        filename4 = os.path.join(output, 'grassVVCOH12_' + year + '.pkl')
        if os.path.exists(filename4):
            grassvv = load_intensities(filename4)
            sns.histplot(grassvv[grassvv > 0], stat='density', ax=axs[1, 0], color=colors[6]).set_title("Grass")

        #filename4 = os.path.join(output, 'stubbleVVCOH12_' + year + '.pkl')
        #if os.path.exists(filename4):
        #    stubblevv = load_intensities(filename4)
        #    sns.histplot(stubblevv[stubblevv > 0], stat='density', ax=axs[1, 1], color=colors[1]).set_title("Stubble")

        filename4 = os.path.join(output, 'stubbleCerealVVCOH12_' + year + '.pkl')
        if os.path.exists(filename4):
            stubbleCerealvv = load_intensities(filename4)
            sns.histplot(stubbleCerealvv[stubbleCerealvv > 0], stat='density', ax=axs[1, 1], color=colors[2]).set_title("Stubble after cereal crop")

        filename4 = os.path.join(output, 'stubbleCompVVCOH12_' + year + '.pkl')
        if os.path.exists(filename4):
            stubbleCompvv = load_intensities(filename4)
            sns.histplot(stubbleCompvv[stubbleCompvv > 0], stat='density', ax=axs[1, 2], color=colors[3]).set_title("Stubble with companion crop")

        fig.suptitle('Distibutions of VV COH12 per class', fontsize=16)
        fig.supxlabel('Coherence')
        print('Saving ' + filename5)
        plt.savefig(os.path.join(output, filename5))
        plt.close()    

        
    if features == ['VV', 'VH']:
        f, ax = plt.subplots(1, 1)
        sns.histplot(populationvv, kde = False, label = 'VV COH12', ax = ax)
        sns.histplot(populationvh, kde = False, label = 'VH COH12', ax = ax)
        ax.set(xlabel='Coherence', ylabel='Counts', title = f'VV and VH COH12 distributions in {year}')
        ax.legend()
        print('Saving ' + os.path.join(output, 'population_distribution.png'))
        plt.tight_layout()        

def readImages(year: str, enddate: str, date: str, parcelpath: str, aoi_shapefile: str, output: str, features, zoneID, savePopulations=False, plotPopulations=False):
        
    myarrays = []
    populationcohvh = []
    populationcohvv = []

    ploughvv = []
    conservationvv = []
    autumnvv = []
    grassvv = []
    stubblevv = []
    stubbleCerealvv = []
    stubbleOthervv = []
    stubbleCompvv = []

    ploughvh = []
    conservationvh = []
    autumnvh = []
    grassvh = []
    stubblevh = []
    stubbleCerealvh = []
    stubbleOthervh = []
    stubbleCompvh = []

    featureset = 'COH12'

    print('Feature set: ' + featureset)
    print('Mosaic year: ' + "".join(list(date)[:4]))
    print('Mosaic starting date: ' + date[4:8])


    head, tail = os.path.split(parcelpath)
    root, ext = os.path.splitext(tail)

    for inSARfilename in glob.glob('/scratch/project_2002224/InSAR/luke/S1A_2023' + enddate + '*COH12.tif'):
        print('\nProcessing InSAR file: ' + inSARfilename.split('/')[-1])

        with rasterio.open(inSARfilename) as insar_raster:
            print('Match projections...')
            # Do InSAR and .shp projections match?
            rasterprojection = insar_raster.crs.to_dict().get('init')
            rasterepsg = rasterprojection[5:] 

            ##reproject the shapefile according to projection of Sentinel2/raster image
            reprojectedshape = os.path.join(head, root + '_reprojected_'+ rasterepsg + ext)
            if not parcelpath.endswith('reprojected_'+rasterepsg+'.shp') and not os.path.exists(reprojectedshape):
                reprojectcommand = 'ogr2ogr -t_srs EPSG:'+rasterepsg+' ' + reprojectedshape + ' ' + parcelpath
                subprocess.call(reprojectcommand, shell=True)
                if not os.path.exists(os.path.splitext(reprojectedshape)[0] +'.prj'):
                    copyfile(os.path.splitext(parcelpath)[0] + '.prj', os.path.splitext(reprojectedshape)[0] +'.prj' )
                print('INFO: ' + parcelpath + ' was reprojected to EPSG code: ' + rasterepsg + ' based on the projection of ' + inSARfilename.split('/')[-1])

            else:
                newparcelpath = parcelpath
                print('INFO: Shapefile was reprojected to EPSG code: ' + rasterepsg + ' based on the projection of ' + inSARfilename.split('/')[-1])


            print(f'Filtering a set of polygons within InSAR bounds...')
            shapes, parcelIDs, partitioning_ix, target = read_shapes_within_bounds(newparcelpath, bb_to_polygon(insar_raster.bounds))
            if len(shapes) == 0:
                print('Found no polygons within InSAR bounds.')
                continue
            else:
                print(f'Found {len(shapes)} polygons within InSAR bounds.  \n')

                gdf = gpd.read_file(newparcelpath)
                gdf2 = gdf[gdf['parcelID'].isin(parcelIDs)]
                #print(gdf2.crs)
                # We need to write this to temp file for zonal_stats:
                gdf2.to_file(os.path.join(newparcelpath[:-4] + '_AOItemp.shp') )
                newparcelpath2 = os.path.join(newparcelpath[:-4] + '_AOItemp.shp')  

                print(f'Extracting pixels...')     

                # Read all pixels within a parcel: 
                feature = 'COH12VV'
                parcelsVV = zonal_stats(newparcelpath2, inSARfilename, band = 2, stats=['count', 'median', 'mean', 'max'], geojson_out=True, all_touched = False, 
                                      raster_out = True, nodata=0) # nodata=-32767
                #print("Length : %d" % len (parcelsVV))

                for x in parcelsVV:
                    myarray = x['properties']['mini_raster_array'].compressed()
                    if not myarray.size:
                        print(f"When reading raster, got empty array for a parcel {x['properties']['parcelID']}.")
                        continue
                    #if x['properties']['max'] > 0:
                    #    print(x['properties'])

                    myid = [x['properties']['parcelID'], x['properties']['target'],
                            x['properties']['count'], x['properties']['mean'], enddate, date, feature]
                    #myid = [x['parcelID'], x['target'], x['split'], 
                    #        x['PINTAALA'], x['properties']['mean'], x['properties']['median'], enddate, feature]
                    #print(myid)
                    arr = myarray.tolist()
                    myid.extend(arr)
                    arr = myid

                    myarrays.append(arr)

                    if savePopulations:
                        populationcohvv.append(myarray.tolist())

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
                        elif x['properties']['target'] == 42:
                            stubbleOthervv.append(myarray.tolist())
                        elif x['properties']['target'] == 5:
                            stubbleCompvv.append(myarray.tolist())
                        else:
                            print(x['properties']['target'])
                            print('Target value did not match any of these: 0, 1, 2, 3, 4, 41, 5')
                            


                feature = 'COH12VH'
                parcelsVH = zonal_stats(newparcelpath2, inSARfilename, band = 1, stats=['count', 'median', 'mean', 'max'], geojson_out=True, all_touched = False, 
                                      raster_out = True, nodata=0) # nodata=-32767
                #print("Length : %d" % len (parcelsVH))

                for x in parcelsVH:
                    myarray = x['properties']['mini_raster_array'].compressed()
                    if not myarray.size:
                        print(f"When reading raster, got empty array for a parcel {x['properties']['parcelID']}.")
                        continue
                    #print(x['properties'])

                    #if x['properties']['max'] > 0:
                    #    print(x['properties'])
                    #else:
                    #    continue

                    myid = [x['properties']['parcelID'], x['properties']['target'],
                            x['properties']['count'], x['properties']['mean'], enddate, date, feature]
                    #myid = [x['parcelID'], x['target'], x['split'], 
                    #        x['PINTAALA'], x['properties']['mean'], x['properties']['median'], enddate, feature]
                    #print(myid)
                    arr = myarray.tolist()
                    myid.extend(arr)
                    arr = myid

                    myarrays.append(arr)

                    if savePopulations:
                        populationcohvh.append(myarray.tolist())

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
                        elif x['properties']['target'] == 42:
                            stubbleOthervh.append(myarray.tolist())
                        elif x['properties']['target'] == 5:
                            stubbleCompvh.append(myarray.tolist())
                        else:
                            print(x['properties']['target'])
                            print('Target value did not match any of these: 0, 1, 2, 3, 4, 41, 5')



    if savePopulations: 
        if ploughvv:
            filename4 = os.path.join(output, 'ploughVVCOH12_' + year + '.pkl')
            save_COH12(filename4, np.concatenate(ploughvv))
        if conservationvv:
            filename4 = os.path.join(output, 'conservationVVCOH12_' + year + '.pkl')
            save_COH12(filename4, np.concatenate(conservationvv))
        if autumnvv:
            filename4 = os.path.join(output, 'autumnVVCOH12_' + year + '.pkl')
            save_COH12(filename4, np.concatenate(autumnvv))
        if grassvv:
            filename4 = os.path.join(output, 'grassVVCOH12_' + year + '.pkl')
            save_COH12(filename4, np.concatenate(grassvv))
        if stubblevv:
            filename4 = os.path.join(output, 'stubbleVVCOH12_' + year + '.pkl')
            save_COH12(filename4, np.concatenate(stubblevv))
        if stubbleCerealvv:
            filename4 = os.path.join(output, 'stubbleCerealVVCOH12_' + year + '.pkl')
            save_COH12(filename4, np.concatenate(stubbleCerealvv))        
        if stubbleOthervv:
            filename4 = os.path.join(output, 'stubbleOtherVVCOH12_' + year + '.pkl')
            save_COH12(filename4, np.concatenate(stubbleOthervv))        
        if stubbleCompvv:
            filename4 = os.path.join(output, 'stubbleCompVVCOH12_' + year + '.pkl')
            save_COH12(filename4, np.concatenate(stubbleCompvv))    

        if ploughvh:
            filename4 = os.path.join(output, 'ploughVHCOH12_' + year + '.pkl')
            save_COH12(filename4, np.concatenate(ploughvh))
        if conservationvh:
            filename4 = os.path.join(output, 'conservationVHCOH12_' + year + '.pkl')
            save_COH12(filename4, np.concatenate(conservationvh))
        if autumnvh:
            filename4 = os.path.join(output, 'autumnVHCOH12_' + year + '.pkl')
            save_COH12(filename4, np.concatenate(autumnvh))
        if grassvh:
            filename4 = os.path.join(output, 'grassVHCOH12_' + year + '.pkl')
            save_COH12(filename4, np.concatenate(grassvh))
        if stubblevh:
            filename4 = os.path.join(output, 'stubbleVHCOH12_' + year + '.pkl')
            save_COH12(filename4, np.concatenate(stubblevh))
        if stubbleCerealvh:
            filename4 = os.path.join(output, 'stubbleCerealVHCOH12_' + year + '.pkl')
            save_COH12(filename4, np.concatenate(stubbleCerealvh))        
        if stubbleOthervh:
            filename4 = os.path.join(output, 'stubbleOtherVHCOH12_' + year + '.pkl')
            save_COH12(filename4, np.concatenate(stubbleOthervh))        
        if stubbleCompvh:
            filename4 = os.path.join(output, 'stubbleCompVHCOH12_' + year + '.pkl')
            save_COH12(filename4, np.concatenate(stubbleCompvh))    

    ###################################################
    # Save all pixel values:     
    filename2 = os.path.join(output, 'fullArrayCOH12' + year + 'InSAR-' + '_'.join(features) + '.pkl')    
    print('Saving all pixels values into ' + filename2)
    save_COH12(filename2, myarrays)

    #print(len(np.concatenate(populationcohvh)), len(np.concatenate(populationcohvv)))

    if savePopulations:

        filename4 = os.path.join(output, 'populationCOH12VVIntensities_' + year + '.pkl')
        save_COH12(filename4, np.concatenate(populationcohvv))        

        filename4 = os.path.join(output, 'populationCOH12VHIntensities_' + year + '.pkl')
        save_COH12(filename4, np.concatenate(populationcohvh))            


    
# HERE STARTS MAIN:

def main(args):

    try:
        if not args.year or not args.outputpath:
            raise Exception('Missing year or output directory arguments. Try --help .')

        print(f'\n01-extractPolygonsFromInSAR.py')
        print(f'\nReading InSAR for the year {args.year}.')
   
        t0 = time.time()

        #nrbins = 16
    
        out_dir_path = Path(os.path.expanduser(args.outputpath + args.year + args.startdate))
        out_dir_path.mkdir(parents=True, exist_ok=True)
        

        
        if args.extractIntensities:
            date, enddate = parseDate(args.startdate, args.year)
            readImages(args.year, enddate, date, args.shppolku, args.aoi_shapefile, out_dir_path, args.alist, args.zonelist, savePopulations=args.savePopulations, plotPopulations=args.plotPopulations)
            
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

    
