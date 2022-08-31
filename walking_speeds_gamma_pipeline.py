#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 17:22:13 2022

@author: James Dowsett
"""


#########  Analysis of walking speeds Gamma   ####################



from flicker_analysis_package import functions

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats
from random import choice
import statistics


### information about the experiment: walking speeds Gamma

path = '/home/james/Active_projects/walking_speeds/subject_data/'

electrode_names = ('O1', 'O2', 'P3', 'P4', 'P7', 'P8', 'HEOG', 'VEOG', 'x_dir', 'y_dir', 'z_dir')

condition_names = ('standing', 'walking slow', 'walking mid', 'walking fast')

location_names = ('hall', 'lobby')

sample_rate = 1000

num_subjects = 25

period = 25

length = 1 # length of FFT in seconds

trig_1_time = -1
trig_2_time = 11
trig_length = 4


#######################################


### Matrices to store results and SSVEPs


all_SSVEPs = np.zeros([num_subjects,2,8,4,25]) # subject, location, electrode, condition, SSVEP data (25 data points = 40 Hz)

SSVEP_amplitudes = np.zeros([num_subjects,2,8,4]) # subject, location, electrode , condition


all_mean_self_absolute_phase_shifts = np.zeros([num_subjects,2,8,4]) # subject, location, electrode , condition

self_split_amplitude_differences = np.zeros([num_subjects,2,8,4]) # subject, location, electrode , condition

self_split_correlation = np.zeros([num_subjects,2,8,4]) # subject, location, electrode , condition

length = 1 # length of FFT in seconds
all_evoked_FFTs = np.zeros([num_subjects,2,8,4,(length * sample_rate)]) # subject, location, electrode, condition, FFT data 


##################################



for subject in range(1,26):
   
    print('  ')
    print('Subject ' + str(subject))
    print(' ')
 
    for location in range(0,2):    
        
        print('Location = ' + location_names[location])
 
    ## load and sort triggers
   
        walking_triggers_file_name = 'S' + str(subject) + '_' + location_names[location] + '_walking_all_triggers.npy'
    
        all_triggers = np.load(path + walking_triggers_file_name)
        
        diff_triggers = np.diff(all_triggers)
        
        # empty lists to put the triggers into for each walking speed
        slow_triggers = []
        mid_triggers = []
        fast_triggers = []
        
        speed = 0 # set to zero to ignore any triggers before the pace markers
        
        for k in range(0,len(diff_triggers)):
                       
            diff_trig = diff_triggers[k]
            # if the triggers are a pace marker, set the walking speed
            if np.abs(diff_trig - 1000) <= 3:
                speed = 1
            elif np.abs(diff_trig - 665) <= 3:
                speed = 2
            elif np.abs(diff_trig - 500) <= 3:
                speed = 3
                
            if np.abs(diff_trig - 25) <= 2: # if triggers are flicker, put into correct list
                
                if speed == 1:
                    slow_triggers.append(all_triggers[k])
                elif speed == 2:
                    mid_triggers.append(all_triggers[k])    
                elif speed == 3:
                    fast_triggers.append(all_triggers[k])    
                    
        print(str(len(slow_triggers)) + ' slow triggers')
        print(str(len(mid_triggers)) + ' mid triggers')
        print(str(len(fast_triggers)) + ' fast triggers')
             
        # convert to numpy arrays
        slow_triggers = np.array(slow_triggers, dtype=int)
        mid_triggers = np.array(mid_triggers, dtype=int)
        fast_triggers = np.array(fast_triggers, dtype=int)
    
        standing_triggers_file_name = 'S' + str(subject) + '_' + location_names[location] + '_standing_all_triggers.npy' 
        standing_triggers = np.load(path + standing_triggers_file_name)
    
        for electrode in range(0,8):
   
        
            electrode_name = electrode_names[electrode]
            
            print(' ')
            print(electrode_name)
            print(' ')


            for condition in range(0,4):
                
                print('\n' + condition_names[condition] + '\n')
            
                ## load raw data
                if condition == 0:
                    data_file_name = 'S' + str(subject) + '_' + location_names[location] + '_standing_chan_' + str(electrode) + '_data.npy'     
                elif condition > 0:
                    data_file_name = 'S' + str(subject) + '_' + location_names[location] + '_walking_chan_' + str(electrode) + '_data.npy'
                
                raw_data = np.load(path + data_file_name)         

        
                if condition == 0:
                    triggers = standing_triggers
                elif condition == 1:
                    triggers = slow_triggers
                elif condition == 2:
                    triggers = mid_triggers
                elif condition == 3:
                    triggers = fast_triggers
                    
                
                ### linear interpolation
                data_linear_interpolation = functions.linear_interpolation(raw_data, triggers, trig_1_time, trig_2_time, trig_length)
                
    
                # make SSVEP
                SSVEP = functions.make_SSVEPs(data_linear_interpolation, triggers, 25) # SIGI was always 40 Hz, length = 25

                 # save SSVEP
                all_SSVEPs[subject-1, location, electrode, condition,:] = SSVEP  # subject, location, electrode, condition, SSVEP data (25 data points = 40 Hz)

                # get absolute self phase shift
                phase_shift = functions.phase_shift_SSVEPs_split(data_linear_interpolation, triggers, period)
                
                all_mean_self_absolute_phase_shifts[subject-1, location, electrode, condition] = phase_shift 

                # get self split amplitude difference
                split_amplitude_difference = functions.SSVEP_split_amplitude_difference(data_linear_interpolation, triggers, period)

                self_split_amplitude_differences[subject-1, location, electrode, condition] = split_amplitude_difference

                # get self correlation
                self_correlation = functions.compare_SSVEPs_split(data_linear_interpolation, triggers, period)

                self_split_correlation[subject-1, location, electrode, condition] = self_correlation 

                ## get Evoked FFT

                
                evoked_FFT = functions.evoked_fft(data_linear_interpolation, triggers, length, sample_rate)
                
                all_evoked_FFTs[subject-1, location, electrode, condition,:] = evoked_FFT # subject, location, electrode, condition, FFT data 




### save 


np.save(path + 'all_SSVEPs', all_SSVEPs) 

np.save(path + 'SSVEP_amplitudes', SSVEP_amplitudes) 


np.save(path + 'all_mean_self_absolute_phase_shifts', all_mean_self_absolute_phase_shifts)
        
np.save(path + 'self_split_amplitude_differences', self_split_amplitude_differences)

np.save(path + 'self_split_correlation', self_split_correlation)
        
np.save(path + 'all_evoked_FFTs', all_evoked_FFTs)
        
        
######
###### Load


all_SSVEPs = np.load(path + 'all_SSVEPs.npy')


SSVEP_amplitudes = np.load(path + 'SSVEP_amplitudes.npy')

all_mean_self_absolute_phase_shifts = np.load(path + 'all_mean_self_absolute_phase_shifts.npy')
 
self_split_amplitude_differences = np.load(path + 'self_split_amplitude_differences.npy')
 
self_split_correlation = np.load(path + 'self_split_correlation.npy')
 
all_evoked_FFTs = np.load(path + 'all_evoked_FFTs.npy')


#############

### plot raw SSVEPs per electrode

electrode_name = 'VEOG'

electrode = electrode_names.index(electrode_name)

for location in range(0,2):
    
    plt.figure()
    plt.suptitle(location_names[location] + ' ' + electrode_name)


    for subject in range(0,25):
        
        plt.subplot(5,5,subject+1)

        plt.title(subject+1)

        for condition in range(0,4):
            
            SSVEP = all_SSVEPs[subject, location, electrode, condition,:]

            plt.plot(SSVEP, label = condition_names[condition])


    plt.legend()






## get phase shifts and Z scores

cc = 0.4 ## correlation cutoff value, reject subjects with self correlation below this value in any condition, 0.4 is 2 SDs

sig_cutoff = 0.05 / 6 # bonforroni corrected for 8 electrodes

for location in range(0,2):
    
    print('\n' + location_names[location] + '\n')
    
    for electrode in range(0,8):
        
            print('\n' + electrode_names[electrode] + '\n')
            
            standing_self_correlations = self_split_correlation[:, location, electrode, 0] 
            slow_self_correlations = self_split_correlation[:, location, electrode, 1] 
            mid_self_correlations = self_split_correlation[:, location, electrode, 2] 
            fast_self_correlations = self_split_correlation[:, location, electrode, 3] 
    
            # only use subjects for which the self correlation is above a certain threshold
            subjects_to_use = []
            for subject in range(0,25):
                if standing_self_correlations[subject] > cc and slow_self_correlations[subject] > cc and mid_self_correlations[subject] > cc and fast_self_correlations[subject] > cc:
                    subjects_to_use.append(subject)

            
            standing_mid_phases_differences = np.zeros([len(subjects_to_use),])
            slow_mid_phases_differences = np.zeros([len(subjects_to_use),])
            fast_mid_phases_differences = np.zeros([len(subjects_to_use),])
            
            mid_self_phase_splits = np.zeros([len(subjects_to_use),])
            
            subject_count = 0

            for subject in subjects_to_use:
                
                standing_SSVEP = all_SSVEPs[subject, location, electrode, 0,:]
                slow_SSVEP = all_SSVEPs[subject, location, electrode, 1,:]
                mid_SSVEP = all_SSVEPs[subject, location, electrode, 2,:]
                fast_SSVEP = all_SSVEPs[subject, location, electrode, 3,:]
              
                standing_mid_phases_differences[subject_count] = functions.cross_correlation_absolute(standing_SSVEP, mid_SSVEP)
                slow_mid_phases_differences[subject_count] = functions.cross_correlation_absolute(slow_SSVEP, mid_SSVEP)
                fast_mid_phases_differences[subject_count] = functions.cross_correlation_absolute(fast_SSVEP, mid_SSVEP)
                
                subject_count += 1
                
            Z_score_standing_mid_vs_slow_mid = functions.group_permutation_test(standing_mid_phases_differences, slow_mid_phases_differences)
            Z_score_fast_mid_vs_slow_mid = functions.group_permutation_test(fast_mid_phases_differences, slow_mid_phases_differences)

            p_value_standing_mid_vs_slow_mid = scipy.stats.norm.sf(abs(Z_score_standing_mid_vs_slow_mid))
            p_value_fast_mid_vs_slow_mid = scipy.stats.norm.sf(abs(Z_score_fast_mid_vs_slow_mid))
            
            average_standing_mid_phase_shift = standing_mid_phases_differences.mean()
            average_slow_mid_phase_shift = slow_mid_phases_differences.mean()
            average_fast_mid_phase_shift = fast_mid_phases_differences.mean()

            print('Average Standing-mid phase shift = ' + str(average_standing_mid_phase_shift))
            print('Average Slow-mid phase shift = ' + str(average_slow_mid_phase_shift))
            print('Average Fast-mid phase shift = ' + str(average_fast_mid_phase_shift))
            
            print('\nZ score standing-mid vs slow-mid =  '  + str(Z_score_standing_mid_vs_slow_mid))
            print('p = ' + str(p_value_standing_mid_vs_slow_mid))
            if p_value_standing_mid_vs_slow_mid < sig_cutoff:
                print('SIGNIFICANT')
            
            print('\nZ score fast-mid vs slow-mid =  '  + str(Z_score_fast_mid_vs_slow_mid))
            print('p = ' + str(p_value_fast_mid_vs_slow_mid))

    
