#
# Adaptation of spontaneous activity in the developing visual cortex
# M. E. Wosniack et al.
# eu
# Data analysis codes
# Auxiliar functions file: extra_functions.py
#
# Author: Marina Wosniack
# Max Planck Institute for Brain Research
# marina-elaine.wosniack@brain.mpg.de
# June 2020
#

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from sklearn.utils import resample
from extra_functions import computing_differences
import scipy
from scipy import stats
# for the statistical tests
from scipy.stats import shapiro # are samples normally distributed?
from scipy.stats import f_oneway # anova... if samples normally distributed
from scipy.stats import kruskal # kruskal-wallis... if samples not normally dist

# defining things for the figures
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 14
def set_style():
    plt.style.use(['seaborn-ticks', 'seaborn-paper'])
    plt.rc('font',  size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes',  titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes',  labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick',  labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick',  labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend',  fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure',  titlesize=BIGGER_SIZE)
sns.set_palette("tab10")
set_style()
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})

#%%
df = pd.read_csv("data/table_siegel_2012.csv")

# preparing the data...
aux_vec_name = df['Animal_name'].to_numpy()
aux_vec_age = df['Animal_age'].to_numpy()
for ii in range(len(aux_vec_name)):
    try:
        len(aux_vec_name[ii])
    except:
        aux_vec_name[ii] = aux_vec_name[ii - 1]
        aux_vec_age[ii] = aux_vec_age[ii - 1]
#
df['Animal_name'] = aux_vec_name
df['Animal_age'] = aux_vec_age
# cleaning more...
df.loc[df['Participation_rate'] > 100, 'Participation_rate'] = 100
df.loc[df['Animal_age'] == 'p8', 'Animal_age'] = 'P8'
df.loc[df['Animal_age'] == 'p9', 'Animal_age'] = 'P9'
df.loc[df['Animal_age'] == 'p10', 'Animal_age'] = 'P10'
#%%
# Setting L or H in the spreadsheet..
aux_vec_event = [None]*len(df)
for ii in range(len(df)):
    if (df['Participation_rate'][ii] >= 20) & (df['Participation_rate'][ii] < 80):
        aux_vec_event[ii] = 'L'
    if (df['Participation_rate'][ii] >= 80):
        aux_vec_event[ii] = 'H'
    if (df['Participation_rate'][ii] < 20):
        aux_vec_event[ii] = 'N'
df['Event_type'] = aux_vec_event
#%%
# Not including here the N-events
data = df[df['Event_type']!='N']
# test if we can do without the age...
data_lh = data.copy(deep=False)
all_names = [individual[:-2] for individual in data_lh['Animal_name']]
data_lh['Individual_name'] = all_names
list_names_files = np.asarray(data_lh['Animal_name'])
list_names_animals = np.asarray(data_lh['Individual_name'])
recording_aux = np.ones(len(data_lh))
list_aux = []
aux_ind = int(list_names_files[0][-2:])
list_aux.append(aux_ind)
for ii in range(1, len(data_lh)):
    list_aux.append(int(list_names_files[ii][-2:]))
data_lh['Recording_number'] = list_aux
#
data_lh.reset_index(drop=True, inplace=True)
#
for ll in range(len(data_lh) - 1):
    if (data_lh['Recording_number'][ll + 1] == data_lh['Recording_number'][ll]) or ((data_lh['Recording_number'][ll + 1] - data_lh['Recording_number'][ll]) == 1):
        recording_aux[ll + 1] = recording_aux[ll]
    else:
        recording_aux[ll + 1] = recording_aux[ll] + 1
#
data_lh['Recording_concat'] = recording_aux
data_lh['Correct_start'] = data_lh['Start_frame']*data_lh['Factor_frame'] + (data_lh['Recording_number'] -1)*300
data_lh['Correct_end'] = data_lh['End_frame']*data_lh['Factor_frame'] + (data_lh['Recording_number'] -1)*300
#%%
print('total recordings: ' + str(len(np.unique(data_lh['Animal_name']))))
print('total animals: ' + str(len(np.unique(data_lh['Individual_name']))))
print('total events: ' + str(len(data_lh)))
print('total H-events: ' + str(len(data_lh[data_lh['Event_type'] == 'H'])))
#
#%%
#
# creating a dataframe to save only the average activity in each recording
data_for_recordings = pd.DataFrame(columns = ['ID', 'Recording', 'Avg_activity'])
for animal in np.unique(data_lh['Individual_name']):
    sub_data = data_lh[data_lh['Individual_name'] == animal]
    for recording in np.unique(sub_data['Recording_number']):
        recording_mean = np.nanmean(sub_data['Amplitude'][sub_data['Recording_number'] == recording])
        data_for_recordings = data_for_recordings.append({'ID':animal, 'Recording':recording, 'Avg_activity':recording_mean}, ignore_index = True)

#%%
# computing now the absolute difference in recording activity for consecutive recordings
amp_diff_recordings = computing_differences(data_for_recordings)
amp_diff_recordings = amp_diff_recordings.abs()
slope, intercept, r_value, p_value, std_err = stats.linregress(x = range(8), y = (amp_diff_recordings).mean())
sns.stripplot( data = amp_diff_recordings, color='tab:gray')
sns.regplot(x = range(8), y = (amp_diff_recordings).mean(), color='r', marker='s', label="y={0:.1e}x+{1:.1e}".format(slope, intercept)).legend(loc="best")
#plt.ylim([-0.01,0.25])
plt.ylabel('Abs. diff. mean recording activities')
plt.xlabel('Recording difference')
sns.despine()
plt.tight_layout()
#plt.savefig('figures/recordings_all_positive.pdf')
# %%
#
# Now lets do some stats on the distributions across diff recordings
#
# 1. removing the NaN to get statistical tests working...
fix_nan_amp_diff = [amp_diff_recordings[col].dropna() for col in amp_diff_recordings]
# 2. now testing if data is normally distributed with the shapiro test
for col in amp_diff_recordings:
    stat, p = shapiro(amp_diff_recordings[col].dropna())
    print(p)
# and data is not normally distributed (p<0.05)
#
# 3. Therefore we do the kruskal test on it
stat, p = kruskal(*fix_nan_amp_diff)
print(p)
# and now since p > 0.05, we get that distributions are probably the same.
#%%
# now the final figure with all the recording events boxplots
dict_lh_anova = {}
for animal in np.unique(data_lh['Individual_name']):
    sub = data_lh[data_lh['Individual_name'] == animal]
    dict_animal = {}
    for recording_num in np.unique(sub['Recording_number']):
        dict_animal[recording_num] = sub[sub['Recording_number'] == recording_num]['Amplitude']
    tst = pd.DataFrame(dict_animal)
    fix_data = [tst[col].dropna() for col in tst]
    stat, p = f_oneway(*fix_data)
    dict_lh_anova[animal] = p
#%%
plt.subplots(7,4, figsize=(20, 17.5))
jj = 1
for animal in np.unique(data_lh['Individual_name']):
    dict_animal = {}
    plt.subplot(7,4,jj)
    sub = data_lh[data_lh['Individual_name'] == animal]
    sns.boxplot(y='Amplitude',x='Recording_number', data = sub, orient='vertical', color='lightblue')
    sns.swarmplot(y='Amplitude',x='Recording_number', data = sub, orient='vertical', color='w', edgecolor='k', linewidth= 1)
    plt.title('ID = ' + animal + ', p = ' + str(round(dict_lh_anova[animal],3)))
    #sns.despine(offset=10, trim=True);
    jj = jj + 1
    sns.despine()
    plt.tight_layout()
plt.savefig('figures/amplitudes_fluctuations.pdf')
