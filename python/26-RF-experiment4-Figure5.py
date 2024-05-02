"""
MY
2023-10-09 / 2023-12-11 for 1240
2024-01-02
2024-03-01

RF importance

RUN: 
python 26-RF-experiment4-Figure5.py

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import seaborn as sns
import pickle




fps = '/Users/myliheik/Documents/myVEGCOVER/vegcover2023/experiment4/featureImportance/ard-S1-S2-2D_2020-2021-2022-2023-InSAR2023-importance-RandomForest-l1-2D-12340.csv'


df = pd.read_csv(fps)
print(f'RF variable importance results from file: {fps}')

# InSAR features are in order:

insarfile = '/Users/myliheik/Documents/myVEGCOVER/vegcover2023/InSAR/20230408/ard2023InSAR.pkl'

def load_InSAR(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    insarfeatures = data.columns.str.replace('bin.*_', '', regex = True)[1:]
    return data, insarfeatures
    
dfInSAR, insarfeatures = load_InSAR(insarfile)

# S1 and S2 features are in order:

features = ['VV', 'VH', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12', 'B8A']
feature_names0 = np.repeat(features, 32)

feature_names = feature_names0.tolist() + insarfeatures.values.tolist()



df['features'] = feature_names
df.rename(columns = {'0': 'importance'}, inplace = True)
df2 = df.set_index('features')
setti = fps.split('-')[11]
df3 = df2.sort_values('importance', ascending = False).reset_index()
print(setti)
df4 = df3.groupby('features').sum().sort_values('importance', ascending = False).reset_index()


dictFeatures = {'B02': 'B02 Blue', 'B03': 'B03 Green', 'B04': 'B04 Red', 'B05': 'B05 Vegetation\nred edge', 'B06': 'B06 Vegetation\nred edge', 'B07': 'B07 Vegetation\nred edge', 
        'B8A': 'B8A Narrow NIR', 'B08': 'B08 NIR', 'B11': 'B11 SWIR', 'B12': 'B12 SWIR', 'VV': 'VV', 'VH': 'VH', 'COH12VV': 'COH12VV', 'COH12VH': 'COH12VH'}

dictSpectralArea = {'B02': 'VIS', 'B03': 'VIS', 'B04': 'VIS', 'B05': 'RENIR', 'B06': 'RENIR', 'B07': 'RENIR', 
        'B8A': 'RENIR', 'B08': 'RENIR', 'B11': 'SWIR', 'B12': 'SWIR', 'VV': 'SAR', 'VH': 'SAR', 'COH12VV': 'InSAR', 'COH12VH': 'InSAR'}


df4['Features'] = df4['features'].map(dictFeatures)
df4['Spectral area'] = df4['features'].map(dictSpectralArea)

print(df4)

def change_width_horizontal(ax, new_value) :
    
    for patch in ax.patches :
        
        current_height = patch.get_height()
        diff = current_height - new_value
        #print(patch.get_y())
        # we change the bar width
        patch.set_height(new_value)

        # we recenter the bar
        patch.set_y(patch.get_y() + diff * .5)
        
        
# colors manually!
fig = plt.figure(figsize=(5, 5), constrained_layout=True)

#figure.set_size_inches(8, 8) # A4 on 8,268 in leveä
fig.set_size_inches(5, 5) # puolikas columnwidth

sns.set_style("whitegrid") # jos haluaa valkoiset taustat

colors = ["skyblue", "green", "red",[0.9312692223325372, 0.8201921796082118, 0.7971480974663592],
           [0.7840440880599453, 0.5292660544265891, 0.6200568926941761],
  [0.402075529973261, 0.23451699199015608, 0.4263168000834109],
         [0.8, 0.47058823529411764, 0.7372549019607844], # B08
  [0.1750865648952205, 0.11840023306916837, 0.24215989137836502],
  [1.0, 0.4980392156862745, 0.054901960784313725],
         [0.9254901960784314, 0.8823529411764706, 0.2]]

colors = ["skyblue", "green", "red", "black"]
          
    
#colors = ["#3182bd", sns.color_palette("tab10")[3], sns.color_palette("tab10")[1], "skyblue"]
colors = [sns.color_palette("Accent")[0],sns.color_palette("Accent")[5], sns.color_palette("tab10")[9],  
          sns.color_palette("tab10")[1], "skyblue"]
#          sns.color_palette("Accent")[1],  "skyblue"]

customPalette = sns.set_palette(sns.color_palette(colors))



ax = sns.barplot(data = df4, y = 'Features', x = 'importance', orient = 'h', hue = 'Spectral area' , palette = customPalette,
                )

plt.legend(loc = "lower right", title = 'Feature set')
#ax.set_xlabel('Relative variable importance')
ax.set_xlabel('Importance score')
#ax.set_ylabel('Count')
#plt.grid(True, alpha=.5) # does not work with eps
plt.grid(True)

change_width_horizontal(ax, .45)

plt.tight_layout()

imgfile = '/Users/myliheik/Documents/myVEGCOVER/vegcover2023/img/article-variableImportance-InSAR.eps'
plt.savefig(imgfile, format = 'eps', dpi = 300)

imgfile = '/Users/myliheik/Dropbox/Apps/Overleaf/Tillage detection for estimating off-season agricultural land-use/img/article-variableImportance-InSAR.eps'
print(f'Saving image to {imgfile}')
plt.savefig(imgfile, format = 'eps', dpi = 300)
imgfile2 = '/Users/myliheik/Dropbox/Apps/Overleaf/Tillage detection for estimating off-season agricultural land-use/img/article-variableImportance-InSAR.png'
print(f'Saving image to {imgfile2}')
plt.savefig(imgfile2, format = 'png', dpi = 300)

    
