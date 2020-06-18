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
from extra_functions import filter_events_time_concat, animals_to_include, exp_decay, make_bootstrap
import scipy
from scipy import stats

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
aux_ind = 1
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
def compute_corr_decay(data_frame, tau_decay, tau_len, threshold_to_include, to_plot=False):
    """
    Main function - returns the correlation statistics for a range of parameters

    Parameters
    ----------
    data_frame: DataFrame
        This is the cleaned data from the siegel spreadsheet
    tau_decay: float
        This is the time constant of the leaky integrator
    tau_len: float
        This is the time window to look back at each H-event
    threshold_to_include: int
        Minimum number of H-events with previous activity to include an animal in the analysis
    to_plot: logic
        If true plots the correlation scatterplot with regression line

    Returns
    -------
    list
        List with the basic statistics from the parameters choice:
        r2: pearson corr coefficient
        pval: pvalue
        C_low: lower bound of 95% CI
        C_top: upper bound of 95% CI
        len(sum_amp_df): number of H-event/prec activity pairs of points in the analysis
        len(selected_animals): number of animals included in the analysis
    """
    clean_data = filter_events_time_concat(data_frame, tau_len)
    selected_animals = animals_to_include(clean_data, threshold_to_include)
    complete_data = data_frame[data_frame['Individual_name'].isin(selected_animals)]
    complete_data = complete_data.reset_index()
    sum_amp_df = pd.DataFrame(columns = ['Individual_name', 'Animal_age', 'Avg_pre_H', 'Amp_H', 'Time_since_last_event', 'Exp_avg_pre_H'])
#
    amps_H = []
    amps_previous = []
    list_name = []
    list_age = []
    list_time_since_event = []
    list_exp_amps_previous = []
#
    for jj in range(len(complete_data)):
        if complete_data['Event_type'][jj] == 'H':
            recording_id = complete_data['Recording_concat'][jj]
            selected_L = complete_data[
                (complete_data['Recording_concat'] == recording_id) &
                (complete_data['Correct_start'][jj] - complete_data['Correct_end'] <= tau_len) &
                (complete_data['Correct_start'][jj] - complete_data['Correct_start'] > 0)]
            if ~np.isnan(np.mean(selected_L['Amplitude'])):
                amps_H.append(complete_data['Amplitude'][jj])
                amps_previous.append(np.mean(selected_L['Amplitude']))
                list_name.append(complete_data['Individual_name'][jj])
                list_age.append(complete_data['Animal_age'][jj])
                list_time_since_event.append(np.min(complete_data['Correct_start'][jj] - selected_L['Correct_end']))
                list_exp_amps_previous.append(
                    np.nanmean  (selected_L['Amplitude']*exp_decay(
                        tau_decay,complete_data['Correct_start'][jj] - selected_L['Correct_end'] )))

    sum_amp_df['Individual_name'] = list_name
    sum_amp_df['Animal_age'] = list_age
    sum_amp_df['Avg_pre_H'] = amps_previous
    sum_amp_df['Exp_avg_pre_H'] = list_exp_amps_previous
    sum_amp_df['Amp_H'] = amps_H
    sum_amp_df['Time_since_last_event'] = list_time_since_event
    sum_amp_df['Index_boot'] = np.arange(0, len(sum_amp_df), 1)
    r2, rpval = scipy.stats.pearsonr(sum_amp_df['Exp_avg_pre_H'], sum_amp_df['Amp_H'])
    # now making the bootstrap:
    C_low, C_top = make_bootstrap(sum_amp_df)
    if to_plot == True:
        r2, rpval = scipy.stats.pearsonr(sum_amp_df['Exp_avg_pre_H'], sum_amp_df['Amp_H'])
        ax = sns.regplot(sum_amp_df['Exp_avg_pre_H'], sum_amp_df['Amp_H'], line_kws={'label':"r={0:.2f}, p={1:.2f}".format(r2,rpval)}, scatter_kws={"s": 50})
        ax.annotate("R2 = {:.2f}".format(r2) + "; p = {:.1e}".format(rpval), xy=(.1, .1), xycoords=ax.transAxes, fontsize='large')
        ax.set_aspect('equal')
        plt.ylim([0.9, 1.6])
        plt.xlim([0.8, 1.5])
        #plt.axis('square')
        ax.set_aspect(np.diff(ax.get_xlim())/np.diff(ax.get_ylim()))
    return [r2, rpval, C_low, C_top, len(sum_amp_df), len(selected_animals)]
#
#%%
# plotting the paper figure
tau_decay = 1000 #decay
tau_len = 300 #len
threshold_to_include = 10
stats = compute_corr_decay(data_lh, tau_decay, tau_len, threshold_to_include, to_plot=True)
plt.savefig('paper_correlation_parameters.pdf')
print('95 % CI: '+str(stats[2])+', '+str(stats[3]))
print('R2: ' + str(stats[0]) + ', ' + 'pval: ' + str(stats[1]))
print('Total points: ' + str(stats[4]) + ', ' + 'total animals: ' + str(stats[5]))
#%%
#
# 1. Calculating the correlation for different event threshold bounds:
#
tau_decay = 1000
tau_len = 300
sum_data_threshold = pd.DataFrame(columns = ['threshold_include', 'r2', 'pval', 'c_min', 'c_max', 'total_points', 'total_animals'])
threshold_len = np.arange(1,24,1)
for threshold_val in threshold_len:
    stats = compute_corr_decay(data_lh, tau_decay, tau_len, threshold_val)
    vec_summary = pd.Series(np.append(threshold_val, stats), index = sum_data_threshold.columns)
    sum_data_threshold = sum_data_threshold.append(vec_summary, ignore_index=True)
#
# figures
#
#%%
plt.plot(sum_data_threshold['threshold_include'], sum_data_threshold['r2'])
plt.fill_between(sum_data_threshold['threshold_include'], sum_data_threshold['c_min'], sum_data_threshold['c_max'], alpha=0.5)
plt.ylim([0,1])
plt.ylabel('Correlation')
plt.xlabel('Threshold events')
#plt.savefig('corr_threshold.pdf')
#%%
plt.plot(sum_data_threshold['threshold_include'], sum_data_threshold['total_animals'],'-o')
plt.ylim([0,26])
plt.xlabel('Threshold to include')
plt.ylabel('Total animals')
#plt.savefig('corr_animals_thresh.pdf')
#%%
#
# 2. Correlation for different time windows tau_len
#
sum_data_tau_len = pd.DataFrame(columns = ['tau_len', 'r2', 'pval', 'c_min', 'c_max', 'total_points', 'total_animals'])
time_window_range = np.arange(75, 570, 12.5)
for time_window in time_window_range:
    stats = compute_corr_decay(data_lh, tau_decay, time_window, threshold_to_include)
    vec_summary = pd.Series(np.append(time_window, stats), index = sum_data_tau_len.columns)
    sum_data_tau_len = sum_data_tau_len.append(vec_summary, ignore_index=True)
#
# figures
#
#%%
plt.plot(sum_data_tau_len['tau_len'], sum_data_tau_len['r2'])
plt.fill_between(sum_data_tau_len['tau_len'], sum_data_tau_len['c_min'], sum_data_tau_len['c_max'], alpha=0.5)
plt.ylim([0,1])
plt.xlabel('Maximum time window')
plt.ylabel('Correlation')
#plt.savefig('corr_time_window_bootstrap.pdf')
#%%
#
# 3. Correlation for different decay time constants
#
sum_data_tau_decay = pd.DataFrame(columns = ['tau_decay', 'r2', 'pval', 'c_min', 'c_max', 'total_points', 'total_animals'])
time_decay_range = np.arange(100,2100,50)
for time_decay in time_decay_range:
    stats = compute_corr_decay(data_lh, time_decay, tau_len, threshold_to_include)
    vec_summary = pd.Series(np.append(time_decay, stats), index = sum_data_tau_decay.columns)
    sum_data_tau_decay = sum_data_tau_decay.append(vec_summary, ignore_index=True)
#
# figures
#
#%%
plt.plot(sum_data_tau_decay['tau_decay'], sum_data_tau_decay['r2'])
plt.fill_between(sum_data_tau_decay['tau_decay'], sum_data_tau_decay['c_min'], sum_data_tau_decay['c_max'], alpha=0.5)
plt.ylim([-0.2,1])
plt.xlabel('Decay time constant')
plt.ylabel('Correlation')
#plt.savefig('corr_decay_bootstrap.pdf')

#%%
