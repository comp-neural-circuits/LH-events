#
# Adaptation of spontaneous activity 2 in the developing visual cortex
# M. E. Wosniack et al.
#
# Data analysis codes
# Auxiliar functions file: extra_functions.py
#
# Author: Marina Wosniack
# Max Planck Institute for Brain Research
# marina-elaine.wosniack@brain.mpg.de
# June 2020
#

import numpy as np
import pandas as pd
from sklearn.utils import resample
import scipy
from scipy import stats

#

def filter_events_time_concat(data_frame, window_length):
    """
    Returns a dataframe with concatenated recordings
    It also calculates the average preceding activity without the leak (not used)
    Used to find which animals satisfy the threshold on number of H-events to
    be included in the analysis

    Parameters
    ----------
    data_frame: DataFrame
        this is the general data frame with events, from the excel spreadsheet
    window_length: float
        maximum window to look back at each H-event to look for previous
        spontaneous activity events. Notice that recordings are concatenated

    Returns
    -------
    dataframe
        contains only H-events and average preceding activity
    """
    amps_H = []
    amps_previous = []
    list_name = []
    list_num_H_window = []
    list_age = []
    list_time_since_event = []
    sum_amp_df = pd.DataFrame(columns = ['Individual_name', 'Animal_age', 'Avg_pre_H', 'Amp_H', 'Time_since_last_event'])
    for kk in range(len(data_frame)):
        if data_frame['Event_type'][kk] == 'H':
            recording_id = data_frame['Recording_concat'][kk]
            selected_L = data_frame[
                (data_frame['Recording_concat'] == recording_id) &
                (data_frame['Correct_start'][kk] - data_frame['Correct_end'] <= window_length) &
                (data_frame['Correct_start'][kk] - data_frame['Correct_start'] > 0) &
                (data_frame['Participation_rate'] >= 20)]
            if ~np.isnan(np.mean(selected_L['Amplitude'])):
                amps_H.append(data_frame['Amplitude'][kk])
                amps_previous.append(np.mean(selected_L['Amplitude']))
                list_name.append(data_frame['Individual_name'][kk])
                list_age.append(data_frame['Animal_age'][kk])
                list_time_since_event.append(np.min(data_frame['Correct_start'][kk] - selected_L['Correct_end']))
    sum_amp_df['Individual_name'] = list_name
    sum_amp_df['Animal_age'] = list_age
    sum_amp_df['Avg_pre_H'] = amps_previous
    sum_amp_df['Amp_H'] = amps_H
    sum_amp_df['Time_since_last_event'] = list_time_since_event
    return(sum_amp_df)

#
def animals_to_include(data_frame, threshold_count):
    """
    Simply checks which animals satisfy the inclusion criteria

    Parameters
    ----------
    data_frame: DataFrame
        this is the output dataframe from the filter_events_time_concat function
    threshold_count: int
        this is the threshold to include an animal in the analysis

    Returns
    -------
    list
        contains the animal IDs to be included in the analysis
    """
    animals_in = []
    for animal_id in np.unique(data_frame['Individual_name']):
        total_H = len(data_frame[data_frame['Individual_name'] == animal_id])
        if total_H >= threshold_count:
            animals_in.append(animal_id)
    return animals_in
#
def exp_decay(tau, time_range):
    """
    A simple exponential decay

    Parameters
    ----------
    tau: float
        decay time constant of the leak integrator
    time_range: float
        time interval to be applied the decay

    Returns
    -------
    float
        exponential decay
    """
    decay = np.exp(- time_range / tau)
    return decay
#
def make_bootstrap(df_boot):
    """
    Bootstrap analysis
    Here I fixed 1000 samples, for the 95% range
    I used the resample function with replacement from sklearn

    Parameters
    ----------
    df_boot: DataFrame
        The dataframe, output of the compute_corr_decay function
    Returns
    -------
    list
        confidence interval (upper and lower)
    """
    total_samples = 1000
    vec_corr_boots = []
    for kk in range(total_samples):
        aux_boot = resample(df_boot['Index_boot'], replace = True, n_samples = len(df_boot))
        r2, rpval = scipy.stats.pearsonr(df_boot['Exp_avg_pre_H'][aux_boot], df_boot['Amp_H'][aux_boot])
        vec_corr_boots.append(r2)
    sorted_vec_corr_boots = np.sort(vec_corr_boots)
    return(sorted_vec_corr_boots[24], sorted_vec_corr_boots[974])
