#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 16:38:22 2022

@author: James Dowsett
"""


#########  Analysis of gamma flicker walk experiment 2   ####################



from flicker_analysis_package import functions

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats
from random import choice
import statistics
import random

from timeit import default_timer as timer
from datetime import timedelta

start_time = timer() # start a timer to keep see how long this is taking

### information about the experiment: Gamma walk 1

path = '/home/james/Active_projects/Gamma_walk/gamma_walk_experiment_2/Right_handed_participants/raw_data_for_analysis_package/'

electrode_names = np.load(path + '/electrode_names.npy')

condition_labels = ['W', 'S', 'black']

condition_names = ['Walking', 'Standing', 'Blackout']

sample_rate = 1000

num_subjects = 24

frequencies_to_use = (35, 40)


trig_1_times = [-1, -1, -1, -1, -1, -1]
trig_2_times = [15, 13, 11, 10, 9, 8]
trig_length = 4

#######################################

### Matrices to store SSVEPs and results of random permutations

# normal reference
all_SSVEPs = np.zeros([num_subjects,2,64,3,29]) # subject, frequency, electrode, condition, SSVEP data (29 data points is the largest SSVEP, 35 Hz)

all_mean_self_correlations = np.zeros([num_subjects,2,64,3])  # subject, frequency, electrode, condition
all_sd_self_correlations = np.zeros([num_subjects,2,64,3])  # subject, frequency, electrode, condition

all_mean_self_phase_shifts = np.zeros([num_subjects,2,64,3])  # subject, frequency, electrode, condition
all_sd_self_phase_shifts = np.zeros([num_subjects,2,64,3])  # subject, frequency, electrode, condition

all_mean_self_absolute_phase_shifts = np.zeros([num_subjects,2,64,3])  # subject, frequency, electrode, condition

all_mean_self_amplitude_differences = np.zeros([num_subjects,2,64,3])  # subject, frequency, electrode, condition
all_sd_self_amplitude_differences = np.zeros([num_subjects,2,64,3])  # subject, frequency, electrode, condition

all_mean_absolute_self_amplitude_differences = np.zeros([num_subjects,2,64,3])  # subject, frequency, electrode, condition

# laplacian reference

all_SSVEPs_laplacian = np.zeros([num_subjects,2,64,3,29]) # subject, frequency, electrode, condition, SSVEP data (29 data points is the largest SSVEP, 35 Hz)

all_mean_self_correlations_laplacian = np.zeros([num_subjects,2,64,3])  # subject, frequency, electrode, condition
all_sd_self_correlations_laplacian = np.zeros([num_subjects,2,64,3])  # subject, frequency, electrode, condition

all_mean_self_phase_shifts_laplacian = np.zeros([num_subjects,2,64,3])  # subject, frequency, electrode, condition
all_sd_self_phase_shifts_laplacian = np.zeros([num_subjects,2,64,3])  # subject, frequency, electrode, condition


all_mean_self_absolute_phase_shifts_laplacian = np.zeros([num_subjects,2,64,3])  # subject, frequency, electrode, condition

all_mean_self_amplitude_differences_laplacian = np.zeros([num_subjects,2,64,3])  # subject, frequency, electrode, condition
all_sd_self_amplitude_differences_laplacian = np.zeros([num_subjects,2,64,3])  # subject, frequency, electrode, condition

all_mean_absolute_self_amplitude_differences_laplacian = np.zeros([num_subjects,2,64,3])  # subject, frequency, electrode, condition

##################################



for laplacian in(0,1): # loop first for normal Cz referance and then again for laplacian
    
    if laplacian == 0:
        montage_name = 'Cz Reference'
    elif laplacian == 1:
        montage_name = 'Laplacian'
    
    print('  ')
    print(montage_name)
    print('  ')
    
    
    for subject in range(1,num_subjects+1):
       
        print('  ')
        print('Subject ' + str(subject))
        print(' ')
     
        conditions_to_use = np.arange(0,3)
        
        if subject == 1 or subject == 16 or subject == 19: # these subjects have no blackout condition
            conditions_to_use = np.array([0, 1])
       
    
        
        for frequency_count in range(0,2):
            
            # plt.figure()
            # plt.suptitle('Subject ' + str(subject) + '  ' + str(frequencies_to_use[frequency_count]) + ' Hz ' + montage_name)
            
            print('Subject ' + str(subject) + '  ' + str(frequencies_to_use[frequency_count]) + ' Hz ' + montage_name)
            

            
            
            frequency = frequencies_to_use[frequency_count]
            
            for condition_count in conditions_to_use:
    
                condition_name = condition_labels[condition_count] + str(frequencies_to_use[frequency_count])
                
    
                
                file_name = 'S' + str(subject) + '_' + condition_name
        
                ## load raw data
                if laplacian == 0:
                    all_data = np.load(path + file_name + '_all_data_interpolated.npy')
                elif laplacian == 1:
                    all_data = np.load(path + file_name + '_all_data_interpolated_laplacian.npy')
                
                ## load triggers
                triggers = np.load(path + file_name + '_all_triggers.npy')
             
                amplitudes_this_condition = np.zeros([64,]) # keep track of the amplitudes for plotting all SSVEPs on the same scale
             
                for electrode in range(0,64):
                    
                    electrode_name = electrode_names[electrode]
                    
                    # print(' ')
                    print(condition_name + ' ' + electrode_name)
                    # print(' ')
                    
                    ### timer
                    elapsed_time_in_seconds = timer() - start_time
                    elapsed_time = timedelta(seconds=elapsed_time_in_seconds)
                    print('Time elapsed = ' + str(elapsed_time))
                    ###
                    
                    
                    # load data for one electrode
                    data = all_data[electrode,:]
                  
                    period = int(np.round(sample_rate/frequency))
                    
                    
                        ### make SSVEP

                    # SSVEP = functions.make_SSVEPs(data, triggers, period)
                    
                    #     # record the SSVEP amplitudes
                    # amplitudes_this_condition[electrode] = np.ptp(SSVEP) # 
                        
                    
                    
                   ###### permutation tests on self correlation, self phase shift and self amplitude difference 
                    num_loops = 1
                    
                    # all_split_corr_values = np.zeros([num_loops,])
                    # all_split_phase_shift_values = np.zeros([num_loops,])
                    
                   # all_split_absolute_phase_shift_values = np.zeros([num_loops,])
                    
                    all_split_amplitude_differences = np.zeros([num_loops,])
                    
                    all_split_absolute_amplitude_differences = np.zeros([num_loops,])
                    
                    for loop in range(0,num_loops):
                        # correlation of random 50/50 split
                        
                       # all_split_corr_values[loop] = functions.compare_SSVEPs_split(data, triggers, period)
                        
                        # phase_shift = functions.phase_shift_SSVEPs_split(data, triggers, period)
                        
                        # # all_split_phase_shift_values[loop] = phase_shift
                        
                        # all_split_absolute_phase_shift_values[loop] = np.abs(phase_shift)
                    
                        split_amplitude_difference = functions.SSVEP_split_amplitude_difference(data, triggers, period)
                        
                        all_split_amplitude_differences[loop] = split_amplitude_difference
                        
                        all_split_absolute_amplitude_differences[loop] = np.abs(split_amplitude_difference)
                    
                    # mean_corr = all_split_corr_values.mean()
                    # sd_corr = np.std(all_split_corr_values)
                    
                    # mean_phase = all_split_phase_shift_values.mean()
                    # sd_phase = np.std(all_split_phase_shift_values)
                    
                   #  mean_abs_phase = all_split_absolute_phase_shift_values.mean()
                    
                    mean_amplitude_difference = all_split_amplitude_differences.mean()
                    std_amplitude_difference = np.std(all_split_amplitude_differences)
                    
                    
                    mean_absolute_amplitude_difference = all_split_absolute_amplitude_differences.mean()
                    
                    ##############
                    
                    # put the SSVEP and the self-permutation test results into the correct matrix
                    if laplacian == 0:

                      #  all_SSVEPs[subject-1,frequency_count,electrode,condition_count,0:period] = SSVEP # subject, frequency, electrode, condition, SSVEP data (29 data points is the largest SSVEP, 35 Hz)
            
                                                
                        # all_mean_self_correlations[subject-1,frequency_count,electrode,condition_count]  = mean_corr # subject, frequency, electrode, condition
                        # all_sd_self_correlations[subject-1,frequency_count,electrode,condition_count]  = sd_corr # subject, frequency, electrode, condition
                        
                        # all_mean_self_phase_shifts[subject-1,frequency_count,electrode,condition_count]  = mean_phase # subject, frequency, electrode, condition
                        # all_sd_self_phase_shifts[subject-1,frequency_count,electrode,condition_count]  = sd_phase # subject, frequency, electrode, condition
                        
                       # all_mean_self_absolute_phase_shifts[subject-1,frequency_count,electrode,condition_count] = mean_abs_phase

                        all_mean_self_amplitude_differences[subject-1,frequency_count,electrode,condition_count] = mean_amplitude_difference
                        all_sd_self_amplitude_differences[subject-1,frequency_count,electrode,condition_count] = std_amplitude_difference

                        all_mean_absolute_self_amplitude_differences[subject-1,frequency_count,electrode,condition_count] = mean_absolute_amplitude_difference

                    elif laplacian == 1:
                        
                       # all_SSVEPs_laplacian[subject-1,frequency_count,electrode,condition_count,0:period] = SSVEP # subject, frequency, electrode, condition, SSVEP data (29 data points is the largest SSVEP, 35 Hz)
            
                   
                        # all_mean_self_correlations_laplacian[subject-1,frequency_count,electrode,condition_count]  = mean_corr # subject, frequency, electrode, condition
                        # all_sd_self_correlations_laplacian[subject-1,frequency_count,electrode,condition_count]  = sd_corr # subject, frequency, electrode, condition
                        
                        # all_mean_self_phase_shifts_laplacian[subject-1,frequency_count,electrode,condition_count]  = mean_phase # subject, frequency, electrode, condition
                        # all_sd_self_phase_shifts_laplacian[subject-1,frequency_count,electrode,condition_count]  = sd_phase # subject, frequency, electrode, condition
                        
                        # all_mean_self_absolute_phase_shifts_laplacian[subject-1,frequency_count,electrode,condition_count] = mean_abs_phase
   
                        all_mean_self_amplitude_differences_laplacian[subject-1,frequency_count,electrode,condition_count] = mean_amplitude_difference
                        all_sd_self_amplitude_differences_laplacian[subject-1,frequency_count,electrode,condition_count] = std_amplitude_difference    
                        
                        all_mean_absolute_self_amplitude_differences_laplacian[subject-1,frequency_count,electrode,condition_count] = mean_absolute_amplitude_difference
   
                    ### plots ###
                    
                #     plt.subplot(8,8,electrode+1)
                    
                #     plt.title(electrode_name)
                    
                #     plt.plot(SSVEP)
                  
                # ##### set constant y axis in plots
                # max_amplitude = np.max(amplitudes_this_condition) 
                # for electrode in range(0,64):  
                #     plt.subplot(8,8,electrode+1)
                #     plt.ylim([-(max_amplitude/2), (max_amplitude/2)])
                
                
                
             
##############  save the resulting matrices  #################            
             
# np.save(path + 'all_SSVEPs', all_SSVEPs)
# np.save(path + 'all_SSVEPs_laplacian', all_SSVEPs_laplacian)                
             

# np.save(path + 'all_mean_self_correlations', all_mean_self_correlations) 
# np.save(path + 'all_sd_self_correlations', all_sd_self_correlations) 

# np.save(path + 'all_mean_self_phase_shifts', all_mean_self_phase_shifts ) 
# np.save(path + 'all_sd_self_phase_shifts', all_sd_self_phase_shifts)

# np.save(path + 'all_mean_self_absolute_phase_shifts', all_mean_self_absolute_phase_shifts)

np.save(path + 'all_mean_self_amplitude_differences', all_mean_self_amplitude_differences)
np.save(path + 'all_sd_self_amplitude_differences', all_sd_self_amplitude_differences)

# np.save(path + 'all_mean_absolute_self_amplitude_differences', all_mean_absolute_self_amplitude_differences)

# np.save(path + 'all_mean_self_correlations_laplacian', all_mean_self_correlations_laplacian) 
# np.save(path + 'all_sd_self_correlations_laplacian', all_sd_self_correlations_laplacian) 

# np.save(path + 'all_mean_self_phase_shifts_laplacian', all_mean_self_phase_shifts_laplacian)
# np.save(path + 'all_sd_self_phase_shifts_laplacian', all_sd_self_phase_shifts_laplacian ) 
   
# np.save(path + 'all_mean_self_absolute_phase_shifts_laplacian', all_mean_self_absolute_phase_shifts_laplacian)

np.save(path + 'all_mean_self_amplitude_differences_laplacian', all_mean_self_amplitude_differences_laplacian)
np.save(path + 'all_sd_self_amplitude_differences_laplacian', all_sd_self_amplitude_differences_laplacian)

# np.save(path + 'all_mean_absolute_self_amplitude_differences_laplacian', all_mean_absolute_self_amplitude_differences_laplacian)
   



########### load the matrices for second part of analysis pipeline ###############

laplacian = 1

if laplacian == 0:
    
    all_SSVEPs = np.load(path + 'all_SSVEPs.npy') # subject, frequency, electrode, condition, SSVEP data (29 data points is the largest SSVEP, 35 Hz)
    
    all_mean_self_correlations = np.load(path + 'all_mean_self_correlations.npy') # subject, frequency, electrode, condition
    all_sd_self_correlations = np.load(path + 'all_sd_self_correlations.npy')
    
    all_mean_self_phase_shifts = np.load(path + 'all_mean_self_phase_shifts.npy')
    all_sd_self_phase_shifts = np.load(path + 'all_sd_self_phase_shifts.npy')    
    
    all_mean_self_absolute_phase_shifts = np.load(path + 'all_mean_self_absolute_phase_shifts.npy')
    
    all_mean_self_amplitude_differences = np.load(path + 'all_mean_self_amplitude_differences.npy')

    all_mean_absolute_self_amplitude_differences = np.load(path + 'all_mean_absolute_self_amplitude_differences.npy')
    
elif laplacian == 1:
    
    all_SSVEPs = np.load(path + 'all_SSVEPs_laplacian.npy')
    
    all_mean_self_correlations = np.load(path + 'all_mean_self_correlations_laplacian.npy')
    all_sd_self_correlations = np.load(path + 'all_sd_self_correlations_laplacian.npy')
    
    all_mean_self_phase_shifts = np.load(path + 'all_mean_self_phase_shifts_laplacian.npy')
    all_sd_self_phase_shifts = np.load(path + 'all_sd_self_phase_shifts_laplacian.npy')    
                
    all_mean_self_absolute_phase_shifts = np.load(path + 'all_mean_self_absolute_phase_shifts_laplacian.npy')         
        
    all_mean_self_amplitude_differences = np.load(path + 'all_mean_self_amplitude_differences_laplacian.npy')
    
    all_mean_absolute_self_amplitude_differences = np.load(path + 'all_mean_absolute_self_amplitude_differences_laplacian.npy')      
             
    
#### Setup Topoplots ########

import mne               
 
### setup montage

file_name = 'S' + str(1) + '_' + 'W35' # load one subjects header file to get the correct montage

raw_data_path = '/home/james/Active_projects/Gamma_walk/gamma_walk_experiment_2/Right_handed_participants/'

# read the EEG data with the MNE function
raw = mne.io.read_raw_brainvision(raw_data_path + file_name + '.vhdr')


channel_names = raw.info.ch_names

channel_names[60] = 'Fpz' # rename incorrectly named channel 

# rename the trigger and EOG as the position of the ground and reference, which were interpolated
raw.info.ch_names[30] =  'FCz' #'A1' 
raw.info.ch_names[31] = 'AFz' # 'A2'  #



sfreq = 1000  # in Hertz

# The EEG channels use the standard naming strategy.
# By supplying the 'montage' parameter, approximate locations
# will be added for them

montage = 'standard_1005'

# Initialize required fields
info = mne.create_info(channel_names, sfreq, ch_types = 'eeg')

info.set_montage(montage)
            


### plot self-correlations 

min_value = 0
max_value = 1

plot_names = ('Walking 35Hz', 'Standing 35 Hz', 'Walking 40 Hz', 'Standing 40Hz')

fig = plt.figure()
plt.suptitle('Mean Self Correlation')


plot_count = 0

for frequency in range(0,2):
    for condition in range(0,2):
        plot_count += 1
        
        plt.subplot(2,2,plot_count)
        
        plt.title(plot_names[plot_count-1])
        
        all_subjects_values = all_mean_self_correlations[:,frequency,:,condition]
      
        
        values_to_plot = all_subjects_values.mean(axis=0)
        
        evoked_values = mne.EvokedArray(np.reshape(values_to_plot, (64,1)), info)

        evoked_values.set_montage(montage)

        mne.viz.plot_topomap(evoked_values.data[:, 0], evoked_values.info,
              vmin=min_value, vmax=max_value, names=channel_names, show_names=True, show=True)

    im,cm = mne.viz.plot_topomap(evoked_values.data[:, 0], evoked_values.info,
  vmin=min_value, vmax=max_value, names=channel_names, show_names=True, show=True)

                
# manually fiddle the position of colorbar
ax_x_start = 0.9
ax_x_width = 0.04
ax_y_start = 0.1
ax_y_height = 0.8
cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
clb = fig.colorbar(im, cax=cbar_ax)




#### get walking-standing phase shift Z scores


min_value = -3  #1.95  # 0  # -2 # 
max_value = 3  #1.96   # 3.9  # 3  # 

fig = plt.figure()
plt.suptitle('Walking Standing Phase Shifts Z scores')



plot_names = ('35 Hz', '40 Hz')


for frequency in range(0,2):
    
    if frequency == 0:
        period = 29
    elif frequency == 1:
        period = 25
  
    plt.subplot(1,2,frequency+1)  
  
    plt.title(plot_names[frequency])
  
    walking_standing_phase_shifts = np.zeros([num_subjects,64])    
    walking_standing_phase_shift_Z_scores = np.zeros([num_subjects,64])      

    walking_amplitudes = np.zeros([num_subjects,64])
    standing_amplitudes = np.zeros([num_subjects,64])
    walking_standing_amplitude_differences = np.zeros([num_subjects,64])     
  
    for subject in range(0,num_subjects):

        for electrode in range(0,64):
            
            walking_SSVEP =  all_SSVEPs[subject,frequency,electrode,0,0:period] # subject, frequency, electrode, condition, SSVEP data
    
            standing_SSVEP =  all_SSVEPs[subject,frequency,electrode,1,0:period]

            phase_shift = functions.cross_correlation_directional(walking_SSVEP, standing_SSVEP)

            abs_phase_shift = np.abs(phase_shift)

            walking_standing_phase_shifts[subject,electrode] = abs_phase_shift

            
            walking_amplitudes[subject,electrode] = np.ptp(walking_SSVEP)
            standing_amplitudes[subject,electrode] = np.ptp(standing_SSVEP)
            
            walking_standing_amplitude_differences[subject,electrode] = np.ptp(walking_SSVEP) - np.ptp(standing_SSVEP)

              ## calculate phase shift Z score for each individual subject from the self 50-50 split phase permutation
              
            # mean_walking_self_phase_shift = all_mean_self_phase_shifts[subject,frequency,electrode,0] # get average mean walking self-phase-shift
            # sd_walking_self_phase_shift = all_sd_self_phase_shifts[subject,frequency,electrode,0] # get average mean walking self-phase-shift

            # phase_shift_Z_score = (phase_shift-mean_walking_self_phase_shift)/(sd_walking_self_phase_shift)

            # abs_phase_shift_Z_score = np.abs(phase_shift_Z_score)

            # walking_standing_phase_shift_Z_scores[subject,electrode] = abs_phase_shift_Z_score


    
    ### get subject level Z score for phase and amplitude
    phase_Z_scores = np.zeros([64,])
    
    amplitude_Z_scores = np.zeros([64,])
    
    for electrode in range(0,64):
        
        ## phase
        true_abs_phase_shifts = walking_standing_phase_shifts[:,electrode]
        self_split_abs_phase_shifts = all_mean_self_absolute_phase_shifts[:,frequency,electrode,0] # load for walking condition
      
        phase_Z_scores[electrode] = functions.group_permutation_test(true_abs_phase_shifts, self_split_abs_phase_shifts)

        ## amplitude
        walking_amplitudes_electrode = walking_amplitudes[:,electrode]
        standing_amplitudes_electrode = standing_amplitudes[:,electrode]
        
        amplitude_Z_scores[electrode] = functions.group_permutation_test(walking_amplitudes_electrode, standing_amplitudes_electrode)

        # Z score from the standard deviation of self split scores
       # Z_scores[electrode] = (true_abs_phase_shifts.mean()-self_split_abs_phase_shifts.mean())/np.std(self_split_abs_phase_shifts)

    # store Z scores for later cluster tests
    if frequency == 0:
        phase_Z_scores_35Hz = np.copy(phase_Z_scores)
        amplitude_Z_scores_35Hz = np.copy(amplitude_Z_scores)
    elif frequency == 1:
        phase_Z_scores_40Hz = np.copy(phase_Z_scores)
        amplitude_Z_scores_40Hz = np.copy(amplitude_Z_scores)

    ### topo-plot ###
    
    #values_to_plot = walking_standing_phase_shifts.mean(axis=0)
    #values_to_plot = walking_standing_phase_shift_Z_scores.mean(axis=0)
    values_to_plot = phase_Z_scores # amplitude_Z_scores   #  
    
    evoked_values = mne.EvokedArray(np.reshape(values_to_plot, (64,1)), info)

    evoked_values.set_montage(montage)

    mne.viz.plot_topomap(evoked_values.data[:, 0], evoked_values.info,
          vmin=min_value, vmax=max_value, names=channel_names, show_names=True, show=True)

    im,cm = mne.viz.plot_topomap(evoked_values.data[:, 0], evoked_values.info,
  vmin=min_value, vmax=max_value, names=channel_names, show_names=True, show=True)

                
# manually fiddle the position of colorbar
ax_x_start = 0.9
ax_x_width = 0.04
ax_y_start = 0.1
ax_y_height = 0.8
cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
clb = fig.colorbar(im, cax=cbar_ax)



############ cluster permutation tests ###############

for frequency in range(0,2):
    
    if frequency == 0:
        Z_scores = np.copy(phase_Z_scores_35Hz)
    elif frequency == 1:
        Z_scores = np.copy(phase_Z_scores_40Hz)
    
    # get adjacency matrix for the EEG electrodes
    adjacency, ch_names = mne.channels.find_ch_adjacency(info, ch_type='eeg')
    
    sig_cutoff = 1.96
    
    true_max_cluster = functions.find_max_cluster(Z_scores, adjacency, ch_names, sig_cutoff)
    
    num_loops = 1000
    
    shuffled_max_cluster_scores = np.zeros([num_loops,])
    
    for loop in range(0,num_loops):
       
        random.shuffle(Z_scores)
        
        shuffled_max_cluster_scores[loop] = functions.find_max_cluster(Z_scores, adjacency, ch_names, sig_cutoff)
        
        
    Z_score_num_clusters = (true_max_cluster - shuffled_max_cluster_scores.mean())/np.std(shuffled_max_cluster_scores)   
    
    
    
    #find p-value
    if frequency == 0:
        p_value_35Hz = scipy.stats.norm.sf(abs(Z_score_num_clusters))
    elif frequency == 1:
        p_value_40Hz = scipy.stats.norm.sf(abs(Z_score_num_clusters))
        
        

print('35 Hz  p value = ' + str(p_value_35Hz))
print('40 Hz  p value = ' + str(p_value_40Hz))





