"""
2021-04-13 MY / 2021-04-23 add mean/median to S1-zonal stats (not tested yet)
2023-01-12 MY bin 32

From histograms to ARD data.

First:
# make ARD per date, all years by default:
python 03-histo2ARD-S1.py -i /Users/myliheik/Documents/myVEGCOVER/vegcover2023/S1/results-insitu -m -l l1

# Finally this combines years, give exact dates: (singleObservations)
python 03-histo2ARD-S1.py -i /Users/myliheik/Documents/myVEGCOVER/vegcover2023/S1/results-insitu -d 20200411 20210411 20220421 -s -l l1

# if non-normalized histogram data needed (e.g. for plotting):
python 03-histo2ARD-S1.py -i /Users/myliheik/Documents/myVEGCOVER/vegcover2023/S1/results-insitu -m




WHERE:

-m combine histograms
#-n combineAnnual2D (intra-annual, all)
-s combineAnnual2D (intra or inter-annual, by dates)
-l normalization, default None

Note: if for some reason there are data from class 4 AND from class 41 and 42 -> duplicates.
it gets rid of class 4! We dont want that. Removed.

"""
import argparse
import numpy as np
import os
import glob
import pandas as pd
from pathlib import Path
import pickle
import textwrap
from sklearn.preprocessing import Normalizer

bin32 = ['bin1', 'bin2', 'bin3', 'bin4', 'bin5', 'bin6', 'bin7', 'bin8', 'bin9', 'bin10', 'bin11', 'bin12', 'bin13', 'bin14', 'bin15', 'bin16',
                         'bin17', 'bin18', 'bin19', 'bin20', 'bin21', 'bin22', 'bin23', 'bin24', 'bin25', 'bin26', 'bin27', 'bin28', 'bin29', 'bin30', 'bin31', 'bin32']
metainfot = ['parcelID', 'target', 'count', 'median', 'startdate', 'date', 'feature']
fullcolumns = metainfot + bin32
    
def save_intensities(filename, arrayvalues):
    with open(filename, 'wb+') as outputfile:
        pickle.dump(arrayvalues, outputfile)

def load_intensities(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def _readFilenames(results_folder_fp, years, keyword):
    inputfiles = []
    # read all files starting with 'histograms'
    for subdir, dirs, files in os.walk(results_folder_fp):
        for file in files:
            if '2018' in years:
                if file.startswith(keyword + '2018'):
                    inputfiles.append(os.path.join(subdir,file))
            if '2019' in years:
                if file.startswith(keyword + '2019'):
                    inputfiles.append(os.path.join(subdir,file))
            if '2020' in years:
                if file.startswith(keyword + '2020'):
                    inputfiles.append(os.path.join(subdir,file))     
            if '2021' in years:
                if file.startswith(keyword + '2021'):
                        inputfiles.append(os.path.join(subdir,file)) 
            if '2022' in years:
                if file.startswith(keyword + '2022'):
                        inputfiles.append(os.path.join(subdir,file)) 
            if '2023' in years:
                if file.startswith(keyword + '2023'):
                        inputfiles.append(os.path.join(subdir,file)) 
    return inputfiles


def make2D(ylist, results_folder_fp, norma):
    print('1. part, makes ARD from histograms.')
    inputfiles = _readFilenames(results_folder_fp, ylist, 'histograms')
    # if for some reason there are data from class 4 AND from class 41 and 42 -> duplicates.
    # let's get rid of class 4:
    for file in inputfiles:
        filename_parts = file.split('S')
        mission = 'S' + filename_parts[1][0]
        
        # outputfile name will have these identifiers:
        year = filename_parts[1][-4:]
        basepath = file.split('/')[:-1]
        print(year)
        
        if mission == 'S1':            
            data = load_intensities(file)
            df = pd.DataFrame(data)
            df.columns = fullcolumns
            print(file)
            print(f"Shape of dataframe: {df.shape}")
            #print(df.iloc[:,7:].head())

            if norma:
                # Normalization:
                dfbins = pd.DataFrame(Normalizer(norm = norma).fit_transform(df.iloc[:,7:]))
            else:
                dfbins = pd.DataFrame(df.iloc[:,7:])
            #print(df.iloc[:,7:].head())
            dfbins.columns = bin32
            df = pd.concat([df.iloc[:,:7], dfbins], axis = 1).copy()
            
            print(len(dfbins), len(df))
            
            # make also median ARD data:
            print(df[['parcelID', 'target', 'count', 'median', 'startdate', 'feature']].head(10))
            pivotedMedian = df[['parcelID', 'target', 'count', 'median', 'startdate', 'feature']].pivot(index=['parcelID'], columns='feature', values=['median'])
            pivotedMedian.columns = pivotedMedian.columns.map('_'.join)
            pivotedMedian = pivotedMedian.reset_index()
            print(pivotedMedian.head())
            print(len(pivotedMedian), len(df))
            
            # Save median:
            outputfile = os.path.join('/'.join(basepath),'median' + year + mission + '.pkl')
            print(f'Saving median data into {outputfile}...')
            save_intensities(outputfile, pivotedMedian)            
            
            # Save meta:
            dfmeta = df[['parcelID', 'target', 'count', 'startdate']].drop_duplicates()
            outputfile = os.path.join('/'.join(basepath),'meta' + year + mission + '.pkl')
            print(f'Saving meta data into {outputfile}...')
            save_intensities(outputfile, dfmeta)
            
            
            pivoted = df.pivot(index=['parcelID'], columns='feature', values=[*df.columns[df.columns.str.startswith('bin')]])
            pivoted.columns = pivoted.columns.map('_'.join)
            pivoted = pivoted.reset_index()
         
            print(f"Shape of dataframe: {pivoted.shape}")
                
            if not norma:
                normaflag = 'not-normalized'
                outputfile = os.path.join('/'.join(basepath),'ard' + year + mission + '.pkl-' + normaflag)
                print(f'Saving histograms into {outputfile}...')
                save_intensities(outputfile, pivoted)            

            else:
                outputfile = os.path.join('/'.join(basepath),'ard' + year + mission + '.pkl')
                print(f'Saving histograms into {outputfile}...')
                save_intensities(outputfile, pivoted)            

def combineAnnual2D(years, inputdir, outputdir, norma): # tätä ei ole testattu vegcover2023

    outputdir = Path(os.path.join(inputdir, 'timeseries/'))
    outputdir.mkdir(parents=True, exist_ok=True)
    print('2. part, makes intra-annual time series into {outputdir}.')
   
    ### Merge all files to one big dataframe
    df_array = []
    meta_array = []
    chosenColumns = ['parcelID', 'target', 'count']
    
    if '2018' in years:
        
        inputfiles = glob.glob(inputdir + '/2018*/ard2018S1.pkl')
        timeseries = pd.concat((iDF.set_index('parcelID') for iDF in map(pd.read_pickle, inputfiles)), sort=False, join='outer', axis=1).reset_index()

        inputfiles = glob.glob(inputdir + '/2018*/meta2018S1.pkl')
        metaseries = pd.concat(map(pd.read_pickle, inputfiles), sort=False, axis=0)
        metaseries2 = metaseries[chosenColumns].drop_duplicates()


        outputfile = os.path.join(outputdir,'ard-' + '2018-' + norma + '.csv.gz')
        print(f'Saving ARD into {outputfile}...')
        timeseries.to_csv(outputfile, index = False)
        
        outputfile = os.path.join(outputdir,'meta-' + '2018-' + norma + '.csv.gz')
        print(f'Saving meta into {outputfile}...')
        metaseries2.to_csv(outputfile, index = False)

    
    if '2019' in years:
        
        inputfiles = glob.glob(inputdir + '/2019*/ard2019S1.pkl')
        timeseries = pd.concat((iDF.set_index('parcelID') for iDF in map(pd.read_pickle, inputfiles)), sort=False, join='outer', axis=1).reset_index()

        inputfiles = glob.glob(inputdir + '/2019*/meta2019S1.pkl')
        metaseries = pd.concat(map(pd.read_pickle, inputfiles), sort=False, axis=0)
        metaseries2 = metaseries[chosenColumns].drop_duplicates()
        
        outputfile = os.path.join(outputdir,'ard-' + '2019-' + norma + '.csv.gz')
        print(f'Saving ARD into {outputfile}...')
        timeseries.to_csv(outputfile, index = False)
        
        outputfile = os.path.join(outputdir,'meta-' + '2019-' + norma + '.csv.gz')
        print(f'Saving meta into {outputfile}...')
        metaseries2.to_csv(outputfile, index = False)

    if '2020' in years:
        
        inputfiles = glob.glob(inputdir + '/2020*/ard2020S1.pkl')
        timeseries = pd.concat((iDF.set_index('parcelID') for iDF in map(pd.read_pickle, inputfiles)), sort=False, join='outer', axis=1).reset_index()

        inputfiles = glob.glob(inputdir + '/2020*/meta2020S1.pkl')
        metaseries = pd.concat(map(pd.read_pickle, inputfiles), sort=False, axis=0)
        metaseries2 = metaseries[chosenColumns].drop_duplicates()
        
        outputfile = os.path.join(outputdir,'ard-' + '2020-' + norma + '.csv.gz')
        print(f'Saving ARD into {outputfile}...')
        timeseries.to_csv(outputfile, index = False)
        
        outputfile = os.path.join(outputdir,'meta-' + '2020-' + norma + '.csv.gz')
        print(f'Saving meta into {outputfile}...')
        metaseries2.to_csv(outputfile, index = False)

    if '2021' in years:
        
        inputfiles = glob.glob(inputdir + '/2021*/ard2021S1.pkl')
        timeseries = pd.concat((iDF.set_index('parcelID') for iDF in map(pd.read_pickle, inputfiles)), sort=False, join='outer', axis=1).reset_index()


        inputfiles = glob.glob(inputdir + '/2021*/meta2021S1.pkl')
        metaseries = pd.concat(map(pd.read_pickle, inputfiles), sort=False, axis=0)
        metaseries2 = metaseries[chosenColumns].drop_duplicates()
        
        outputfile = os.path.join(outputdir,'ard-' + '2021-' + norma + '.csv.gz')
        print(f'Saving ARD into {outputfile}...')
        timeseries.to_csv(outputfile, index = False)
        
        outputfile = os.path.join(outputdir,'meta-' + '2021-' + norma + '.csv.gz')
        print(f'Saving meta into {outputfile}...')
        metaseries2.to_csv(outputfile, index = False)
 
    if '2022' in years:
        
        inputfiles = glob.glob(inputdir + '/2022*/ard2022S1.pkl')
        timeseries = pd.concat((iDF.set_index('parcelID') for iDF in map(pd.read_pickle, inputfiles)), sort=False, join='outer', axis=1).reset_index()


        inputfiles = glob.glob(inputdir + '/2022*/meta2022S1.pkl')
        metaseries = pd.concat(map(pd.read_pickle, inputfiles), sort=False, axis=0)
        metaseries2 = metaseries[chosenColumns].drop_duplicates()
        
        outputfile = os.path.join(outputdir,'ard-' + '2022-' + norma + '.csv.gz')
        print(f'Saving ARD into {outputfile}...')
        timeseries.to_csv(outputfile, index = False)
        
        outputfile = os.path.join(outputdir,'meta-' + '2022-' + norma + '.csv.gz')
        print(f'Saving meta into {outputfile}...')
        metaseries2.to_csv(outputfile, index = False)

    if '2023' in years:
        
        inputfiles = glob.glob(inputdir + '/2023*/ard2023S1.pkl')
        timeseries = pd.concat((iDF.set_index('parcelID') for iDF in map(pd.read_pickle, inputfiles)), sort=False, join='outer', axis=1).reset_index()


        inputfiles = glob.glob(inputdir + '/2023*/meta2023S1.pkl')
        metaseries = pd.concat(map(pd.read_pickle, inputfiles), sort=False, axis=0)
        metaseries2 = metaseries[chosenColumns].drop_duplicates()
        
        outputfile = os.path.join(outputdir,'ard-' + '2023-' + norma + '.csv.gz')
        print(f'Saving ARD into {outputfile}...')
        timeseries.to_csv(outputfile, index = False)
        
        outputfile = os.path.join(outputdir,'meta-' + '2023-' + norma + '.csv.gz')
        print(f'Saving meta into {outputfile}...')
        metaseries2.to_csv(outputfile, index = False)
        
def combineSingle2D(dates, inputdir, outputdir, norma):


    outputdir = Path(os.path.join(inputdir, 'combined')) 
    outputdir.mkdir(parents=True, exist_ok=True)
    
    print('3. part, combines files from different years (= unique parcelIDs). Saves into {outputdir}.')
    
    ### Merge all files to one big dataframe
    df_array = []
    median_array = []
    years = []

    for date in dates:

        year = date[:4]
        years.append(year)

        inputfile = os.path.join(inputdir + '/' + date + '/ard' + year + 'S1.pkl')
        print(inputfile)
        # Read ARD file:
        df = pd.read_pickle(inputfile)
        print(len(df))
        #list_of_names = df.columns
        #features = ['_'.join(e.split('_')[:-1]) for e in list_of_names[1:]]
        #newcolumns = ['parcelID']
        #newcolumns.extend(features)
        #df = df.set_axis(newcolumns, axis=1)

        #print(df.head())

        # Read all meta files:
        inputfile = os.path.join(inputdir + '/' + date + '/median' + year + 'S1.pkl')
        median = pd.read_pickle(inputfile)
        median_array.append(median)
        
        # Read all meta files:
        inputfile = os.path.join(inputdir + '/' + date + '/meta' + year + 'S1.pkl')
        metaseries = pd.read_pickle(inputfile)
        metaseries2 = metaseries[['parcelID', 'target', 'count', 'startdate']].drop_duplicates() # mediaania ei voi ottaa tähän, koska siitä on merkattu vv ja vh mediaani, joten tulee duplikaatteja

        df_array.append(df.merge(metaseries2))

    # Join:
    dfAll = pd.concat(df_array, axis=0, ignore_index=True)
    outputfile = os.path.join(outputdir,'ard-' + '_'.join(dates) + '-' + norma + '.csv.gz')
    print(f'Saving ARD into {outputfile}...')
    dfAll.to_csv(outputfile, index = False)
 
    dfAll2 = pd.concat(median_array, axis=0, ignore_index=True)
    outputfile = os.path.join(outputdir,'median-' + '_'.join(dates) + '-' + norma + '.csv.gz')
    print(f'Saving median ARD into {outputfile}...')
    dfAll2.to_csv(outputfile, index = False)


    
# MAIN:

def main(args):

    try:
        if not args.inputpath:
            raise Exception('Missing input directory arguments. Try --help .')

        print(f'\n03-histo2ARD-S1.py')
        print(f'\nReading histograms for the year(s) {", ".join(args.ylist)}.')
   
        if not args.outputpath:
            outputpath = args.inputpath
        else:
            outputpath = args.outputpath
            
        #years = ['2018', '2019']
        years = args.ylist

        out_dir_path = Path(os.path.expanduser(outputpath))
        out_dir_path.mkdir(parents=True, exist_ok=True)
        
        inputdirpaths = glob.glob(args.inputpath + '/20*/')
        
        for inputdirpath in inputdirpaths:
        
            if args.make2D:
                make2D(years, inputdirpath, args.normalization)
        

        if args.combineAnnual2D:
            combineAnnual2D(years, args.inputpath, out_dir_path, args.normalization)
            
        if args.combineSingle2D:
            combineSingle2D(args.datelist, args.inputpath, out_dir_path, args.normalization)
            
        
        print(f'\nDone.')
            
    except Exception as e:
        print('\n\nUnable to read input or write out results. Check prerequisites and see exception output below.')
        parser.print_help()
        raise e
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=textwrap.dedent(__doc__))

    parser.add_argument('-i', '--inputpath',
                        type = str,
                        help = 'Upper level directory for histograms files.')
    parser.add_argument('-y', '--years', action='store', dest='ylist',
                       type=str, nargs='*', default=['2018', '2019', '2020', '2021', '2022', '2023'],
                       help="Optionally e.g. -y 2018 2019, default all")
    parser.add_argument('-o', '--outputpath',
                        type=str,
                        help='Directory for output files. Default: input directory')
    parser.add_argument('-m', '--make2D',
                        help='Combine histogram files.',
                        default=False,
                        action='store_true')
    parser.add_argument('-n', '--combineAnnual2D',
                        help="Combine histograms of features into one big dataframe annual.",
                        default=False,
                        action='store_true')  
    parser.add_argument('-l', '--normalization',
                        type=str, 
                        default=None, 
                        help='Type of normalization (l1 or l2). Default: None')
    parser.add_argument('-s', '--combineSingle2D',
                        help="Combine histograms of features into one big dataframe of single window observations by given dates.",
                        default=False,
                        action='store_true')  
    parser.add_argument('-d', '--dates', action='store', dest='datelist',
                       type=str, nargs='*', default=None,
                       help="Give dates e.g. -d 20200411 20210411")

    parser.add_argument('--debug',
                        help='Verbose output for debugging.',
                        action='store_true')

    args = parser.parse_args()
    main(args)

