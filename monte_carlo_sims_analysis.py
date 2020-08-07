#
# Adaptation of spontaneous activity 2 in the developing visual cortex
# M. E. Wosniack et al.
#
# Data analysis codes
#
# This combines the information of Monte Carlo simulations
# where I changed the learning rule (Hebbian, BCM), properties
# of H-events (adaptive, not-adaptive), and properties of L-events
# (I changed their sizes).
#
# The range of input threshold is [0.3, 0.45] for Hebbian with non-adaptive
# H-events and [0.3, 0.7] for Hebbian with adaptive H-events. This range
# was chosen such that we have the three possible simulation outcomes within it
# non-selective, selective and decoupled receptive fields (though RFs do not
# decouple without H-events or with adaptive H-events)
#
# The range of target rates of the BCM rule was chosen due to the same reason
#
# The range of H-event properties that I explore resembles the observations in
# Siegel et al 2012.
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

# reading the data...
data_BCM = pd.read_csv('data/data_BCM_mc.csv')
data_BCM_only_L = pd.read_csv('data/data_BCM_only_L_mc.csv')
data_not_adaptive = pd.read_csv('data/data_not_adaptive_mc.csv')
data_not_adaptive_only_L = pd.read_csv('data/data_not_adaptive_only_L_mc.csv')
data_adaptive = pd.read_csv('data/data_adaptive_mc.csv')
data_adaptive_diff_L = pd.read_csv('data/data_adaptive_diff_L_mc.csv')

#%%
#
# RF size MC simulations Hebbian rule non adaptive H-events
fig, ax = plt.subplots()
plt.scatter(data_not_adaptive['theta_u'], data_not_adaptive['Hint'], c=data_not_adaptive['Size'], cmap = 'viridis',s=50,vmin=0, vmax=1)
plt.xlim([0.3,0.45])
plt.ylim([2, 5])
plt.colorbar()
plt.xlabel('theta_u')
plt.ylabel('Hint')
ax.set_aspect(1./ax.get_data_ratio())
plt.savefig('figures/scatter_not_adaptive_size.pdf')
#
#%%
#
# To supp materials... sparseness of the Hebbian rule not-adaptive
fig, ax = plt.subplots()
plt.scatter(data_not_adaptive['theta_u'], data_not_adaptive['Hint'], c=data_not_adaptive['Sparsity'], cmap = 'viridis',s=50,vmin=0, vmax=1)
plt.xlim([0.3,0.45])
plt.ylim([2, 5])
plt.colorbar()
plt.xlabel('theta_u')
plt.ylabel('Hint')
ax.set_aspect(1./ax.get_data_ratio())
plt.savefig('figures/scatter_not_adaptive_sparseness.pdf')
#%%
#
# RF size MC simulations Hebbian rule non adaptive H-events
fig, ax = plt.subplots()
plt.scatter(data_not_adaptive_only_L['theta_u'], data_not_adaptive_only_L['Max_L'], c=data_not_adaptive_only_L['Size'], cmap = 'viridis',s=50,vmin=0, vmax=1)
plt.xlim([0.3,0.7])
plt.ylim([0.2, 1])
plt.colorbar()
plt.xlabel('theta_u')
plt.ylabel('Lmax')
ax.set_aspect(1./ax.get_data_ratio())
plt.savefig('figures/scatter_not_adaptive_only_L_size.pdf')


#%%
#
# RF size MC simulations BCM learning rule
fig, ax = plt.subplots()
plt.scatter(data_BCM['y0'], data_BCM['Hint'], c=data_BCM['Size'], cmap = 'viridis',s=50,vmin=0, vmax=1)
plt.xlim([0.4,1.2])
plt.ylim([2, 5])
plt.colorbar()
plt.xlabel('y0')
plt.ylabel('Hint')
ax.set_aspect(1./ax.get_data_ratio())
plt.savefig('figures/scatter_BCM_size.pdf')
#
#%%
# This will go into supplementary materials. The sparseness of the BCM
fig, ax = plt.subplots()
plt.scatter(data_BCM['y0'], data_BCM['Hint'], c=data_BCM['Sparsity'], cmap = 'viridis',s=50,vmin=0, vmax=1)
plt.xlim([0.4,1.2])
plt.ylim([2, 5])
plt.colorbar()
plt.xlabel('y0')
plt.ylabel('Hint')
ax.set_aspect(1./ax.get_data_ratio())
plt.savefig('figures/scatter_BCM_sparseness.pdf')
#%%
#%%
#
# RF size MC simulations Hebbian rule with adaptive H-events
fig, ax = plt.subplots()
plt.scatter(data_adaptive['theta_u'], data_adaptive['Hint'], c=data_adaptive['Size'], cmap = 'viridis',s=50,vmin=0, vmax=1)
plt.xlim([0.3,0.7])
plt.ylim([2, 5])
plt.colorbar()
plt.xlabel('theta_u')
plt.ylabel('Hint')
ax.set_aspect(1./ax.get_data_ratio())
plt.savefig('figures/scatter_adaptive_size.pdf')
#%%
#
# RF size MC simulations Hebbian rule with adaptive H-events
# Here Hint = 3.5 (fixed) and I varied the maximum size of the L-events
# The idea is that smaller L-events lead to smaller receptive field sizes
# and better topography
fig, ax = plt.subplots()
plt.scatter(data_adaptive_diff_L['theta_u'], data_adaptive_diff_L['Max_L'], c=data_adaptive_diff_L['Size'], cmap = 'viridis',s=50,vmin=0, vmax=1)
plt.xlim([0.3,0.7])
plt.ylim([0.2,1])
plt.colorbar()
plt.xlabel('theta_u')
plt.ylabel('Lmax')
ax.set_aspect(1./ax.get_data_ratio())
plt.savefig('figures/scatter_adaptive_diff_L_size.pdf')
#%%
#
# Now the percentages of each type of simulation result... for Hebbian rule non-adaptive H-events
splot = sns.barplot(x='Classification', y='Classification', data=data_not_adaptive, palette='Blues', order=['S', 'NS', 'D'],estimator=lambda x: len(x) / len(data_not_adaptive) * 100, orient='v')
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.ylabel('Classification (%)')
plt.ylim([0, 100])
plt.savefig('figures/counts_not_adaptive.pdf')
#%%
#
# BCM barplot
splot = sns.barplot(x='Classification', y='Classification', data=data_BCM, palette='Blues', order=['S', 'NS', 'D'],estimator=lambda x: len(x) / len(data_BCM) * 100, orient='v')
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.ylabel('Classification (%)')
plt.ylim([0, 100])
plt.savefig('figures/counts_BCM.pdf')

#%%
#
# Hebbian rule with adaptive H-events barplot
splot = sns.barplot(x='Classification', y='Classification', data=data_adaptive, palette='Blues', order=['S', 'NS', 'D'],estimator=lambda x: len(x) / len(data_adaptive) * 100, orient='v')
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.ylabel('Classification (%)')
plt.ylim([0, 100])
plt.savefig('figures/counts_adaptive.pdf')
#%%
#
# comparing BCM properties with only L-events
# The idea is to show that the topography of the BCM rule is better for smaller
# target rates, but those good target rates do not generate refined receptive fields
# in the case with H-events
bins = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
data_BCM_only_L_selective = data_BCM_only_L[data_BCM_only_L['Classification'] == 'S']
data_BCM_only_L_selective['y0_range'] = pd.cut(data_BCM_only_L_selective['y0'], bins)

f, ax = plt.subplots()
sns.despine(bottom=True, left=True)
sns.despine(bottom=True, left=True)

sns.stripplot( 'Topography', 'y0_range', data=data_BCM_only_L_selective, dodge=True, alpha=.25, zorder=1)
sns.pointplot('Topography','y0_range',  data=data_BCM_only_L_selective, dodge=.532, join=False, palette="dark", markers="d", scale=.75)
handles, labels = ax.get_legend_handles_labels()
plt.ylabel('y0')
plt.xlim([0,1])
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('figures/topo_y0_BCM.pdf')
#%%
# comparing BCM properties with only L-events
bins = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
data_BCM_only_L['y0_range'] = pd.cut(data_BCM_only_L['y0'], bins)

f, ax = plt.subplots()
sns.despine(bottom=True, left=True)
sns.despine(bottom=True, left=True)

sns.stripplot( 'Size', 'y0_range', data=data_BCM_only_L, dodge=True, alpha=.25, zorder=1)
sns.pointplot('Size','y0_range',  data=data_BCM_only_L, dodge=.532, join=False, palette="dark", markers="d", scale=.75)
handles, labels = ax.get_legend_handles_labels()
plt.ylabel('y0')
plt.xlim([0,1])
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('figures/size_y0_BCM.pdf')

#%%
#
# comparing topography values for the Hebbian rule with non-adaptive H-events
bins = [0.2, 0.4, 0.6, 0.8, 1.0]
data_not_adaptive_only_L_selective = data_not_adaptive_only_L[data_not_adaptive_only_L['Classification'] == 'S']
data_not_adaptive_only_L_selective['Size_range'] = pd.cut(data_not_adaptive_only_L_selective['Max_L'], bins)

f, ax = plt.subplots()
sns.despine(bottom=True, left=True)
sns.despine(bottom=True, left=True)

sns.stripplot( 'Topography', 'Size_range', data=data_not_adaptive_only_L_selective, dodge=True, alpha=.25, zorder=1)
sns.pointplot('Topography','Size_range',  data=data_not_adaptive_only_L_selective, dodge=.532, join=False, palette="dark", markers="d", scale=.75)
handles, labels = ax.get_legend_handles_labels()
plt.ylabel('Maximum L-event Size (%)')
plt.xlim([0,1])
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('figures/topo_size_L_non_adapt.pdf')

#%%
#
# now showing the dependence on RF size (non-adaptive H-events, Hebbian rule)
bins = [0.2, 0.4, 0.6, 0.8, 1.0]
data_not_adaptive_only_L['Size_range'] = pd.cut(data_not_adaptive_only_L['Max_L'], bins)
f, ax = plt.subplots()
sns.despine(bottom=True, left=True)
sns.despine(bottom=True, left=True)

sns.stripplot( 'Size', 'Size_range', data=data_not_adaptive_only_L, dodge=True, alpha=.25, zorder=1)
sns.pointplot('Size','Size_range',  data=data_not_adaptive_only_L, dodge=.532, join=False, palette="dark", markers="d", scale=.75)
handles, labels = ax.get_legend_handles_labels()
plt.ylabel('Maximum L-event Size (%)')
plt.tight_layout()
plt.savefig('figures/rfsize_size_L_non_adapt.pdf')
#%%
#
# Now going to the Hebbian rule with adaptive H-events
bins = [0.2, 0.4, 0.6, 0.8, 1.0]
data_adaptive_diff_L_selective = data_adaptive_diff_L[data_adaptive_diff_L['Classification'] == 'S']
data_adaptive_diff_L_selective['Size_range'] = pd.cut(data_adaptive_diff_L_selective['Max_L'], bins)

f, ax = plt.subplots()
sns.despine(bottom=True, left=True)
sns.despine(bottom=True, left=True)

sns.stripplot( 'Topography', 'Size_range', data=data_adaptive_diff_L_selective, dodge=True, alpha=.25, zorder=1)
sns.pointplot('Topography','Size_range',  data=data_adaptive_diff_L_selective, dodge=.532, join=False, palette="dark", markers="d", scale=.75)
handles, labels = ax.get_legend_handles_labels()
plt.ylabel('Maximum L-event Size (%)')
plt.xlim([0,1])
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('figures/topo_size_L_adapt.pdf')


#%%
#
# now showing the dependence on RF size
bins = [0.2, 0.4, 0.6, 0.8, 1.0]
data_adaptive_diff_L['Size_range'] = pd.cut(data_adaptive_diff_L['Max_L'], bins)
f, ax = plt.subplots()
sns.despine(bottom=True, left=True)
sns.despine(bottom=True, left=True)

sns.stripplot( 'Size', 'Size_range', data=data_adaptive_diff_L, dodge=True, alpha=.25, zorder=1)
sns.pointplot('Size','Size_range',  data=data_adaptive_diff_L, dodge=.532, join=False, palette="dark", markers="d", scale=.75)
handles, labels = ax.get_legend_handles_labels()
plt.ylabel('Maximum L-event Size (%)')
plt.gca().invert_yaxis()
plt.xlim([0,1])
plt.tight_layout()
plt.savefig('figures/rfsize_size_L_adapt.pdf')

#%%
# Comparing now the topography of receptive fields
# attention: here with H-events, all the rules have the same range of H-events!!!
# Also attention, we are sub-sampling the data to study the topography of only the
# refined receptive fields
#
not_adaptive_topo_selective = data_not_adaptive['Topography'][data_not_adaptive['Classification']=='S']
BCM_topo_selective = data_BCM['Topography'][data_BCM['Classification']=='S']
adaptive_topo_selective = data_adaptive['Topography'][data_adaptive['Classification'] == 'S']
combined_df_topo = pd.DataFrame({'not_adaptive':pd.Series(not_adaptive_topo_selective), 'BCM':pd.Series(BCM_topo_selective), 'adaptive':pd.Series(adaptive_topo_selective)})
plt.subplots(figsize = (7,5))
sns.boxplot(data = combined_df_topo, palette='Blues')
sns.stripplot(data = combined_df_topo, color='w', jitter=0.2)
plt.ylabel('Topography')
plt.savefig('figures/all_topo.pdf')
#%%
#
# Sparseness of the selective receptive fields
not_adaptive_spars_selective = data_not_adaptive['Sparsity'][data_not_adaptive['Classification']=='S']
BCM_spars_selective = data_BCM['Sparsity'][data_BCM['Classification']=='S']
adaptive_spars_selective = data_adaptive['Sparsity'][data_adaptive['Classification'] == 'S']
combined_df_spars = pd.DataFrame({'not_adaptive':pd.Series(not_adaptive_spars_selective), 'BCM':pd.Series(BCM_spars_selective), 'adaptive':pd.Series(adaptive_spars_selective)})
plt.subplots(figsize = (7,5))
sns.barplot(data = combined_df_spars, palette='Blues')
#sns.stripplot(data = combined_df_spars, color='w', jitter=0.2)
plt.ylabel('Sparseness')
plt.savefig('figures/all_sparseness.pdf')
#%%
#
# Reporting mean and STD for the topography values:
print('Hebbian not adaptive: mean T = ' + str(round(np.mean(not_adaptive_topo_selective),2)) + ' STD = ' + str(round(np.std(not_adaptive_topo_selective),2)) )
print('BCM: mean T = ' + str(round(np.mean(BCM_topo_selective),2)) + ' STD = ' + str(round(np.std(BCM_topo_selective),2)) )
print('Hebbian adaptive: mean T = ' + str(round(np.mean(adaptive_topo_selective),2)) + ' STD = ' + str(round(np.std(adaptive_topo_selective),2)) )

#%%
print('Hebbian not adaptive: mean S = ' + str(round(np.mean(not_adaptive_spars_selective),2)) + ' STD = ' + str(round(np.std(not_adaptive_spars_selective),2)) )
print('BCM: mean S = ' + str(round(np.mean(BCM_spars_selective),2)) + ' STD = ' + str(round(np.std(BCM_spars_selective),2)) )
print('Hebbian adaptive: mean S = ' + str(round(np.mean(adaptive_spars_selective),2)) + ' STD = ' + str(round(np.std(adaptive_spars_selective),2)) )
