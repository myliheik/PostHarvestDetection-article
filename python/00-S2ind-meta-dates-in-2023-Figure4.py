"""
2024-03-18 MY

Best pixel based S2 indices, what are the dates?

April 2023 -> DOYs 91...120

Meta dates can be found in: /scratch/project_2002224/S2mosaics/pta_sjp_s2ind_meta_20230401_20230430.tif

RUN:
python 00-S2ind-meta-dates-in-2023.py

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

from scipy import stats
import itertools

import time

colors = ["black", "goldenrod" , "saddlebrown", "gold", "skyblue", "yellowgreen", "darkseagreen", "darkgoldenrod"]

savein = '/scratch/project_2002224/vegcover2023/img/S2ind-range-20-80-dates-300dpi.png'
savein2 = '/scratch/project_2002224/vegcover2023/img/S2ind-all-dates-300dpi.png'
saveineps = '/scratch/project_2002224/vegcover2023/img/S2ind-range-20-80-dates-300dpi.eps'
savein2eps = '/scratch/project_2002224/vegcover2023/img/S2ind-all-dates-300dpi.eps'

fp = '/scratch/project_2002224/S2mosaics/pta_sjp_s2ind_meta_20230401_20230430.tif'
shpfile = '/projappl/project_2002224/vegcover2023/shpfiles/insitu/extendedReferences2023-AOI-sampled-buffered.shp'

try:
    if not os.path.exists(shpfile):
        raise Exception('Missing Shapefile with parcel polygons.')

    parcelpath = shpfile

    print(f'Parcel path is {parcelpath}.')

except Exception as e:
    print('\n\nUnable to read shapefile for parcels.')
    #parser.print_help()
    raise e

# Read all pixels within a parcel:
parcels = zonal_stats(shpfile, fp, stats=['count', 'min', 'max', 'percentile_20', 'percentile_80'], geojson_out=True, all_touched = False, 
                      raster_out = True, nodata=65535)
print("Length : %d" % len(parcels))

populationdates = []
myarrays = []
i = 0
for x in parcels:
    myarray = x['properties']['mini_raster_array'].compressed()
    if not myarray.size:
        i = i + 1
        continue
    myid = x['properties']['percentile_80'] - x['properties']['percentile_20']

    populationdates.append(myarray.tolist())

    myarrays.append(myid)

populationdates2 = list(itertools.chain(*populationdates))
print(f'Statistics on all dates in April 2023: {stats.describe(populationdates2)}')

print(f'Statistics on all dates in April 2023: {stats.describe(myarrays)}')

print(f"When reading raster, got empty array for {i} out of {len(parcels)} parcels ({round(i/len(parcels), 2)}%).")

# how many doys are between April 11-21 (article):
j = [i for i in myarrays if i >= 102 and i <= 111]

print(f"How many doys are between April 11-21? {len(j)}, ({round(len(j)/len(myarrays), 3)}%)")

### Range dates:

##sns.set_style('darkgrid')
#sns.set_style("whitegrid")

sns.set_style('whitegrid', rc={
    'xtick.bottom': True,
    'ytick.left': False,
})

fig = plt.figure(figsize = (2.8, 2.5))
##ax = fig.add_axes([0, 0, 1, 1])
plt.rcParams.update({'font.size': 8})
#rect = plt.Rectangle((101, 0), 10, 0.255,
#                     facecolor="yellow", alpha=0.1)

g = sns.histplot(myarrays, stat='density', color=colors[5], bins = 22)
#g.add_patch(rect)
#plt.axvline(101, color="k", linestyle="--");
#plt.axvline(111, color="k", linestyle="--");

g.set_xlabel('Δ days')

g.set_xticks([0.5, 5.5, 10.5, 15.5, 20.5], labels = [0, 5, 10, 15, 20])
print(f'Saving plot as {savein} ...')
plt.tight_layout()
plt.savefig(savein, dpi = 300)
plt.savefig(saveineps, dpi = 300)


### All dates:

#fig = plt.figure(figsize=[6, 2.5])
fig = plt.figure(figsize = (2.8, 2.5))
#ax = fig.add_axes([0, 0, 1, 1])
plt.rcParams.update({'font.size': 8})
rect = plt.Rectangle((101, 0), 10, 0.255,
                     facecolor="yellow", alpha=0.1)

g = sns.histplot(populationdates2, stat='density', color=colors[5], bins = 27)
g.add_patch(rect)
plt.axvline(101, color="k", linestyle="--");
plt.axvline(111, color="k", linestyle="--");
g.set_xlabel('Date')

g.set_xticks([91.5, 96.5, 99.5, 101.5, 104.5, 106.5, 111.5, 114.5, 117.5], labels = ['April 1', 'April 6', 'April 9', 'April 11', 'April 14', 'April 16', 'April 21', 'April 24', 'April 27'], rotation=45)
print(f'Saving plot as {savein2} ...')
plt.tight_layout()
plt.savefig(savein2, dpi = 300)
plt.savefig(savein2eps, dpi = 300)


print(f'Done.')
