"""
2021-04-13 MY
2024-01-01 MY 

From histograms to ARD data. One time point only, multiple observations
-> Apply add/sum of duplicates per window, i.e. merge all observations per window per parcel into one. Window is 12-days COH.

python 03-histo2ARD-InSAR.py -i /Users/myliheik/Documents/myVEGCOVER/vegcover2023/InSAR -y 2023 -m -l l1

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
    print('Makes ARD from histograms.')
    inputfiles = _readFilenames(results_folder_fp, ylist, 'histograms')


    # if for some reason there are data from class 4 AND from class 41 and 42 -> duplicates.
    # let's get rid of class 4:
    for file in inputfiles:
        filename_parts = file.split('I')

        mission = 'InSAR'
        
        # outputfile name will have these identifiers:
        year = filename_parts[1][-4:]
        basepath = file.split('/')[:-1]

        data = load_intensities(file)
        df00 = pd.DataFrame(data)
        df00.columns = fullcolumns
        print(df00)
        # Drop median:
        df = df00.drop('median', axis = 1)
        print(f"Shape of dataframe: {df.shape}")
        print(df.iloc[:,6:].head())

        df2 = df[['parcelID', 'feature'] +  df.columns[df.columns.str.startswith('bin')].tolist()].groupby(['parcelID', 'feature']).aggregate(np.sum).reset_index()

        #df2 = df[['parcelID', 'feature'] +  df.columns[df.columns.str.startswith('bin')].tolist()]
        print(df2.head(2))
        print(f"Any duplicates? {df2.duplicated(subset=['parcelID', 'feature']).sum()}")
        print(df2.iloc[:,2:].head())
        if norma:
            # Normalization:
            dfbins = pd.DataFrame(Normalizer(norm = norma).fit_transform(df2.iloc[:,2:]))
        else:
            dfbins = pd.DataFrame(df2.iloc[:,2:])
        #print(df.iloc[:,6:].head())
        dfbins.columns = bin32
        df3 = pd.concat([df2.iloc[:,:2], dfbins], axis = 1).copy()
        print(df3.head(2))

        print(len(dfbins), len(df3))
        
        #df3['parcelIDPolar'] = df3['parcelID'] + df3['feature']
        
        #print(df3[df3['parcelIDPolar'].duplicated()])

        # make also median ARD data:
        #print(df3[['parcelID', 'target', 'count', 'median', 'startdate', 'feature']].head(10))
        #pivotedMedian = df3[['parcelID', 'target', 'count', 'median', 'startdate', 'feature']].pivot(index=['parcelID'], columns='feature', values=['median'])
        #pivotedMedian.columns = pivotedMedian.columns.map('_'.join)
        #pivotedMedian = pivotedMedian.reset_index()
        #print(pivotedMedian.head())
        #print(len(pivotedMedian), len(df))

        # Save median:
        #outputfile = os.path.join('/'.join(basepath),'median' + year + mission + '.pkl')
        #print(f'Saving median data into {outputfile}...')
        #save_intensities(outputfile, pivotedMedian)            

        # Save meta:
        dfmeta = df3[['parcelID', 'feature']]
        outputfile = os.path.join('/'.join(basepath),'meta' + year + mission + '.pkl')
        print(f'Saving meta data into {outputfile}...')
        save_intensities(outputfile, dfmeta)


        pivoted = df3.pivot(index=['parcelID'], columns='feature', values=[*df3.columns[df3.columns.str.startswith('bin')]])
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


    
# MAIN:

def main(args):

    try:
        if not args.inputpath:
            raise Exception('Missing input directory arguments. Try --help .')

        print(f'\n03-histo2ARD-InSAR.py')
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
        print(inputdirpaths)
        for inputdirpath in inputdirpaths:
        
            if args.make2D:
                make2D(years, inputdirpath, args.normalization)
               
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

