#
# Adaptation of spontaneous activity 2 in the developing visual cortex
# M. E. Wosniack et al.
#
# Data analysis code for the effect of L-events in the BCM learning rule
# Requires data output from MATLAB simulations
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
style.use('seaborn-white')
sns.set_palette("colorblind")
from scipy.io import loadmat # reading matlab files
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
set_style()
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})

#%%
# loading .mat file data
data_L_events_load = loadmat('data/L-details.mat')
# selecting the arrays to be used
L_size = data_L_events_load['record_L_sizes']
output_L = data_L_events_load['record_L_output']
theta_L = data_L_events_load['record_L_theta']
# transforming L_sizes in %
L_size_pct = (L_size/50)*100
#%%
# first figure, scatterplot with a cortical cell's output, its current
# dynamic threshold value, color-coded with the event size
# We have that events registered above the diagonal lead to LTP
# Events below the diagonal lead to LTD
#%%
f, ax = plt.subplots()
cell_num = 0 # we can sample any cell, from 0 to 49
num_events = len(output_L[cell_num][output_L[cell_num] > 0]) # only events that triggered LTP or LTD in this cell
#sub_sample_events = np.arange(int(num_events/2), num_events) # lets use fewer points so the figure is not so messy
sub_sample_events = np.arange(0, int(num_events/4))  # lets use fewer points so the figure is not so messy
act_L = output_L[cell_num][output_L[cell_num] > 0][sub_sample_events]
threshold_L = theta_L[cell_num][output_L[cell_num] > 0][sub_sample_events]
size_L = L_size_pct.ravel()[output_L[cell_num] > 0][sub_sample_events]
#
plt.scatter(threshold_L, act_L, c=size_L, vmin=20, vmax=80, s=40, cmap='Blues')
plt.colorbar()
plt.ylim([0,20])
plt.xlim([0,20])
ax.plot([0,20], [0,20], ls="--", c=".3")
plt.xlabel('Sliding threshold')
plt.ylabel('L-event output')
ax.set_aspect(1./ax.get_data_ratio())
plt.savefig('figures/scatter_L_sizes.pdf')
#%%
# The figure above was just a qualitative argument to why only large L-events
# lead to LTP in the BCM rule. Now we quantify it at the population level to
# get the histograms
#
#%%
# find the events that actually induce LTP or LTD in the BCM rule
# remember that one need coincident pre and post-activity to modify
# synaptic weights here
potentiates = output_L - theta_L
sizes_potentiates = [] # will store the size of potentiating events
sizes_depresses = [] # will store the size of depressive events
sizes_nothing = [] # not used
for ii in range(50): # because we have 50 cortical cells
    aux_potentiates = np.where((potentiates[ii] > 0) & (output_L[ii] > 0)) # positive pre and positive post - thresh
    aux_depresses = np.where((potentiates[ii] < 0) & (output_L[ii] > 0)) # positive pre and negative post - thresh
    aux_nothing = np.where(potentiates[ii] == 0) # not used
    # now saving everybody in the same pack
    sizes_potentiates.append(L_size_pct[aux_potentiates].ravel())
    sizes_depresses.append(L_size_pct[aux_depresses].ravel())
    sizes_nothing.append(L_size_pct[aux_nothing].ravel())
#%%
#
# just need to flatten the data to a single long vector
flat_depresses = [item for sublist in sizes_depresses for item in sublist]
flat_potentiates = [item for sublist in sizes_potentiates for item in sublist]
flat_nothing = [item for sublist in sizes_nothing for item in sublist]
#%%
# here we will plot the histograms of events that potentiate or depress
plt.hist(flat_depresses, bins=[20, 30, 40, 50, 60, 70, 80], histtype='bar', ec='k', density = False)
plt.xlabel('Event size')
plt.title('Events that depress')
plt.xlim([20, 80])
plt.ylim([0, 36000]) # correcting the limits by hand
plt.ylabel('Counts')
#%%
# same for the potentiating events
plt.hist(flat_potentiates, bins = [20, 30, 40, 50, 60, 70, 80], histtype='bar', ec='k', density = False)
plt.xlabel('Event size')
plt.title('Events that potentiate')
plt.xlim([20, 80])
plt.ylim([0, 36000]) # correcting the limits by hand
plt.ylabel('Counts')
#%%
# just confirming that we generated events of all the sizes uniformly
#
plt.hist(flat_depresses + flat_potentiates + flat_nothing, bins=[20, 30, 40, 50, 60, 70, 80], histtype='bar', ec='k', density = False)
plt.xlabel('Event size')
plt.title('All events')
plt.xlim([20,80])
plt.ylabel('Counts')
#%%
