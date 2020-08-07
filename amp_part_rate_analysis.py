# Data analysis codes
#
# This extracts the amplitude vs participation rate relationships
# from the model and the data
#
# The goal is to show the flattening of the plots when we increase the
# input threshold of the learning rule, which is a parallel for increasing
# the age of the animal
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
from scipy import stats

# defining things for the figures
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 14
def set_style():
    plt.style.use(['seaborn-ticks', 'seaborn-paper'])
    #plt.rc("font", family="Helvetica")
    plt.rc('font',  size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes',  titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes',  labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick',  labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick',  labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend',  fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure',  titlesize=BIGGER_SIZE)
set_style()
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})

#%%
#
# First reading the experimental data and cleaning it
df = pd.read_csv("data/table_siegel_2012.csv")
df = df[df['Participation_rate'] >= 20]
df.loc[df['Participation_rate'] > 100, 'Participation_rate'] = np.nan
#
#%%
# Getting the info from the data...
plt.plot(df['Participation_rate'], df['Amplitude'], '.', color = 'gray')
slope, intercept, r_value, p_value, std_err = stats.linregress(df['Participation_rate'][df['Participation_rate']>=80], df['Amplitude'][df['Participation_rate']>=80])
ax=sns.regplot(df['Participation_rate'][df['Participation_rate']>=80], df['Amplitude'][df['Participation_rate']>=80],color='r', scatter=False, label="y={0:.1e}x+{1:.1f}".format(slope, intercept))
ax.legend()
plt.ylim([0.8,1.6])
plt.xlim([20,100])
sns.despine()
plt.tight_layout()
plt.savefig('figures/data_amp_part.pdf')
#%%


# Now the model data
df_model_045 = pd.read_csv('data/amp_part_rate_model_045.csv')
df_model_050 = pd.read_csv('data/amp_part_rate_model_050.csv')
df_model_060 = pd.read_csv('data/amp_part_rate_model_060.csv')
#%%
list_part_rates = np.arange(0,20,2)
list_amps = np.arange(1,20,2)
#
slope_list_045 = []
plt.subplots(2, 5, figsize = (22, 6), sharex = True, sharey = True)
for ii in range(10):
    plt.subplot(2, 5, ii + 1)
    aux_ind = df_model_045.iloc[:,list_part_rates[ii]] > 80
    plt.plot(df_model_045.iloc[:,list_part_rates[ii]], df_model_045.iloc[:,list_amps[ii]], '.', color = 'gray')
    slope, intercept, r_value, p_value, std_err = stats.linregress(df_model_045.iloc[:, list_part_rates[ii]][aux_ind],df_model_045.iloc[:, list_amps[ii]][aux_ind])
    slope_list_045.append(slope)
    sns.regplot(df_model_045.iloc[:, list_part_rates[ii]][aux_ind],df_model_045.iloc[:, list_amps[ii]][aux_ind], color='r', scatter=False, label="y={0:.1e}x+{1:.1f}".format(slope, intercept))
    sns.despine()
    plt.legend()
    plt.ylim([0,12])
    plt.xlim([20,100])
    plt.tight_layout()
plt.savefig('figures/amp_part_045.pdf')
#%%
#
slope_list_050 = []
plt.subplots(2, 5, figsize = (22, 7), sharex = True, sharey = True)
for ii in range(10):
    plt.subplot(2, 5, ii + 1)
    aux_ind = df_model_050.iloc[:,list_part_rates[ii]] > 80
    plt.plot(df_model_050.iloc[:,list_part_rates[ii]], df_model_050.iloc[:,list_amps[ii]], '.', color = 'gray')
    slope, intercept, r_value, p_value, std_err = stats.linregress(df_model_050.iloc[:, list_part_rates[ii]][aux_ind],df_model_050.iloc[:, list_amps[ii]][aux_ind])
    slope_list_050.append(slope)
    sns.regplot(df_model_050.iloc[:, list_part_rates[ii]][aux_ind],df_model_050.iloc[:, list_amps[ii]][aux_ind], color='r', scatter=False, label="y={0:.1e}x+{1:.1f}".format(slope, intercept))
    sns.despine()
    plt.legend()
    plt.ylim([0,12])
    plt.xlim([20,100])
    plt.tight_layout()
plt.savefig('figures/amp_part_050.pdf')

#%%
#
slope_list_060 = []
plt.subplots(2, 5, figsize = (22, 7), sharex = True, sharey = True)
for ii in range(10):
    plt.subplot(2, 5, ii + 1)
    aux_ind = df_model_060.iloc[:,list_part_rates[ii]] > 80
    plt.plot(df_model_060.iloc[:,list_part_rates[ii]], df_model_060.iloc[:,list_amps[ii]], '.', color = 'gray')
    slope, intercept, r_value, p_value, std_err = stats.linregress(df_model_060.iloc[:, list_part_rates[ii]][aux_ind],df_model_060.iloc[:, list_amps[ii]][aux_ind])
    slope_list_060.append(slope)
    sns.regplot(df_model_060.iloc[:, list_part_rates[ii]][aux_ind],df_model_060.iloc[:, list_amps[ii]][aux_ind], color='r', scatter=False, label="y={0:.1e}x+{1:.1f}".format(slope, intercept))
    sns.despine()
    plt.legend()
    plt.ylim([0,12])
    plt.xlim([20,100])
    plt.tight_layout()
plt.savefig('figures/amp_part_060.pdf')

#%%
Dict_slopes = pd.DataFrame(columns = ['theta_045', 'theta_050', 'theta_060'])
Dict_slopes['theta_045'] = slope_list_045
Dict_slopes['theta_050'] = slope_list_050
Dict_slopes['theta_060'] = slope_list_060
sns.boxplot(data = Dict_slopes, palette = 'Oranges')
sns.stripplot(data = Dict_slopes, color='tab:gray', s=8)
plt.ylabel('H-events slope')
sns.despine()
plt.savefig('figures/slopes_H_events_boxplot.pdf')
#%%
sns.barplot(data = Dict_slopes, palette = 'Oranges', orient = 'h')
plt.ylabel('H-events slope')
sns.despine()
plt.savefig('figures/slopes_H_events_barplot.pdf')
#%%
# Reporting the stats
print('Mean 045: ' + str(np.mean(slope_list_045)) + ', STD: ' + str(np.std(slope_list_045)))
print('Mean 050: ' + str(np.mean(slope_list_050)) + ', STD: ' + str(np.std(slope_list_050)))
print('Mean 060: ' + str(np.mean(slope_list_060)) + ', STD: ' + str(np.std(slope_list_060)))
