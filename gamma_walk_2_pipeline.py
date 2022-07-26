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


# laplacian reference

all_SSVEPs_laplacian = np.zeros([num_subjects,2,64,3,29]) # subject, frequency, electrode, condition, SSVEP data (29 data points is the largest SSVEP, 35 Hz)

all_mean_self_correlations_laplacian = np.zeros([num_subjects,2,64,3])  # subject, frequency, electrode, condition
all_sd_self_correlations_laplacian = np.zeros([num_subjects,2,64,3])  # subject, frequency, electrode, condition

all_mean_self_phase_shifts_laplacian = np.zeros([num_subjects,2,64,3])  # subject, frequency, electrode, condition
all_sd_self_phase_shifts_laplacian = np.zeros([num_subjects,2,64,3])  # subject, frequency, electrode, condition

##################################





for laplacian in(0,1):
    
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
                  
            
                        ### make SSVEP
                
                    period = int(np.round(sample_rate/frequency))
                    
                    SSVEP = functions.make_SSVEPs(data, triggers, period)
                    
                        # record the SSVEP amplitudes
                    amplitudes_this_condition[electrode] = np.ptp(SSVEP) # 
                        
                    
                    
                   ###### permutation tests on self correlation and phase shift
                    num_loops = 1000
                    
                    all_split_corr_values = np.zeros([num_loops,])
                    all_split_phase_shift_values = np.zeros([num_loops,])
                    
                    for loop in range(0,num_loops):
                        # correlation of random 50/50 split
                        
                        all_split_corr_values[loop] = functions.compare_SSVEPs_split(data, triggers, period)
                        
                        all_split_phase_shift_values[loop] = functions.phase_shift_SSVEPs_split(data, triggers, period)
                        
                    
                    mean_corr = all_split_corr_values.mean()
                    sd_corr = np.std(all_split_corr_values)
                    
                    mean_phase = all_split_phase_shift_values.mean()
                    sd_phase = np.std(all_split_phase_shift_values)
                    
                    ##############
                    
                    # put the SSVEP and the self-permutation test results into the correct matrix
                    if laplacian == 0:

                        all_SSVEPs[subject-1,frequency_count,electrode,condition_count,0:period] = SSVEP # subject, frequency, electrode, condition, SSVEP data (29 data points is the largest SSVEP, 35 Hz)
            
                                                
                        all_mean_self_correlations[subject-1,frequency_count,electrode,condition_count]  = mean_corr # subject, frequency, electrode, condition
                        all_sd_self_correlations[subject-1,frequency_count,electrode,condition_count]  = sd_corr # subject, frequency, electrode, condition
                        
                        all_mean_self_phase_shifts[subject-1,frequency_count,electrode,condition_count]  = mean_phase # subject, frequency, electrode, condition
                        all_sd_self_phase_shifts[subject-1,frequency_count,electrode,condition_count]  = sd_phase # subject, frequency, electrode, condition
                        


                    elif laplacian == 1:
                        all_SSVEPs_laplacian[subject-1,frequency_count,electrode,condition_count,0:period] = SSVEP # subject, frequency, electrode, condition, SSVEP data (29 data points is the largest SSVEP, 35 Hz)
            
                   
                        all_mean_self_correlations_laplacian[subject-1,frequency_count,electrode,condition_count]  = mean_corr # subject, frequency, electrode, condition
                        all_sd_self_correlations_laplacian[subject-1,frequency_count,electrode,condition_count]  = sd_corr # subject, frequency, electrode, condition
                        
                        all_mean_self_phase_shifts_laplacian[subject-1,frequency_count,electrode,condition_count]  = mean_phase # subject, frequency, electrode, condition
                        all_sd_self_phase_shifts_laplacian[subject-1,frequency_count,electrode,condition_count]  = sd_phase # subject, frequency, electrode, condition
                        
                        
                    
                    ### plots ###
                    
                #     plt.subplot(8,8,electrode+1)
                    
                #     plt.title(electrode_name)
                    
                #     plt.plot(SSVEP)
                  
                # ##### set constant y axis in plots
                # max_amplitude = np.max(amplitudes_this_condition) 
                # for electrode in range(0,64):  
                #     plt.subplot(8,8,electrode+1)
                #     plt.ylim([-(max_amplitude/2), (max_amplitude/2)])
             
                
             
np.save(path + 'all_SSVEPs', all_SSVEPs)
np.save(path + 'all_SSVEPs_laplacian', all_SSVEPs_laplacian)                
             

np.save(path + 'all_mean_self_correlations', all_mean_self_correlations) 
np.save(path + 'all_sd_self_correlations', all_sd_self_correlations) 

np.save(path + 'all_mean_self_phase_shifts', all_mean_self_phase_shifts ) 
np.save(path + 'all_sd_self_phase_shifts', all_sd_self_phase_shifts)


np.save(path + 'all_mean_self_correlations_laplacian', all_mean_self_correlations_laplacian) 
np.save(path + 'all_sd_self_correlations_laplacian', all_sd_self_correlations_laplacian) 

np.save(path + 'all_mean_self_phase_shifts_laplacian', all_mean_self_phase_shifts_laplacian)
np.save(path + 'all_sd_self_phase_shifts_laplacian', all_sd_self_phase_shifts_laplacian ) 
   
                
         ####### matricies to store the analysis outcomes     
   
# ## normal reference
             
# SSVEP_amplitudes = np.zeros([num_subjects,2,64,3]) # subject, frequency, electrode , condition

# SSVEP_walking_standing_correlations = np.zeros([num_subjects,2,64]) # subject, frequency, electrode

# SSVEP_walking_standing_phase_shift = np.zeros([num_subjects,6,8])             
                
#          ### laplacian
         
# SSVEP_amplitudes_laplacian = np.zeros([num_subjects,2,64,3]) # subject, frequency, electrode , condition

# SSVEP_walking_standing_correlations_laplacian = np.zeros([num_subjects,2,64]) # subject, frequency, electrode

# SSVEP_walking_standing_phase_shift_laplacian = np.zeros([num_subjects,6,8])


   

                
             