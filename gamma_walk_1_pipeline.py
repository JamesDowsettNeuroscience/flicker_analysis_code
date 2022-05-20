#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 12:55:09 2022

@author: James Dowsett
"""

#########  Analysis of gamma flicker walk experiment 1   ####################



from flicker_analysis_package import functions

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats
from random import choice
import statistics


### information about the experiment: Gamma walk 1

path = '/home/james/Active_projects/Gamma_walk/Gamma_walking_experiment_1/raw_data_for_analysis_package/'

electrode_names = ('VEOG', 'blink', 'P3', 'P4', 'O2', 'Pz', 'O1', 'HEOG', 'x_dir', 'y_dir', 'z_dir')

condition_names = ('standing', 'walking')

sample_rate = 1000

num_subjects = 10

frequencies_to_use = (30, 35, 40, 45, 50, 55)

trig_1_times = [-1, -1, -1, -1, -1, -1]
trig_2_times = [15, 13, 11, 10, 9, 8]
trig_length = 3

#######################################


SIGI_amplitudes = np.zeros([num_subjects,6,8,2]) # subject, frequency electrode, condition

SIGI_walking_standing_correlations = np.zeros([num_subjects,6,8])

SIGI_phase_scores = np.zeros([num_subjects,6,8])


SSVEP_amplitudes = np.zeros([num_subjects,6,8,2]) # subject, frequency, electrode , condition

SSVEP_walking_standing_correlations = np.zeros([num_subjects,6,8]) # subject, frequency, electrode

SSVEP_phase_scores = np.zeros([num_subjects,6,8])




for subject in range(1,11):
   
    print('  ')
    print('Subject ' + str(subject))
    print(' ')
    # plt.figure()
    # plt.suptitle(subject)
    
    for electrode in range(0,8):
        
        electrode_name = electrode_names[electrode]
        
        print(' ')
        print(electrode_name)
        print(' ')
        
        ## load data
        data_file_name = 'subject_' + str(subject) + '_electrode_' + str(electrode) + '_data.npy'
        
        data = np.load(path + data_file_name)                



        ####### SIGI conditions
        
        frequency_count = 0
        for frequency in frequencies_to_use:

            for condition in range(0,2):
                
                ## load triggers from real SSVEP condition to match the number of triggers to use
                triggers_file_name = 'subject_' + str(subject) + '_' + condition_names[condition] + '_' + str(frequency) + 'Hz_triggers.npy'   
                triggers = np.load(path + triggers_file_name)    
            
                num_triggers_to_use = len(triggers)
                
                
                # load the SIGI triggers
                triggers = np.load(path + 'subject_' + str(subject) + '_SIGI_' + condition_names[condition] + '_triggers.npy')
                
                # only use the same number of triggers that there were in the real SSVEP condition
                triggers = triggers[0:num_triggers_to_use]
                
                print(condition_names[condition] + ' ' + str(len(triggers)))
                
                ### make SSVEP
                
                period = int(np.round(sample_rate/40))
                
                SSVEP = functions.make_SSVEPs(data, triggers, period)
                
                
                
               # plt.plot(SSVEP)
        
                SIGI_amplitudes[subject-1,frequency_count,electrode,condition] = np.ptp(SSVEP)
        
                if condition == 0:
                    standing_SSVEP = np.copy(SSVEP)
                elif condition== 1:
                    walking_SSVEP = np.copy(SSVEP)
        
        
            SIGI_walking_standing_correlations[subject-1,frequency_count,electrode] = np.corrcoef(standing_SSVEP, walking_SSVEP)[0,1]
        
            SIGI_phase_scores[subject-1,frequency_count,electrode] = functions.cross_correlation(standing_SSVEP, walking_SSVEP)
        
            frequency_count += 1
            
            
    
        ######### make real SSVEPs  ########################
        frequency_count = 0
        for frequency in frequencies_to_use:
            for condition in range(0,2):
            
                plt.subplot(2,3,frequency_count+1)    
            
                ## load triggers
                triggers_file_name = 'subject_' + str(subject) + '_' + condition_names[condition] + '_' + str(frequency) + 'Hz_triggers.npy'
                
                triggers = np.load(path + triggers_file_name)
                
                ### linear interpolation
                data_linear_interpolation = functions.linear_interpolation(data, triggers, trig_1_times[frequency_count], trig_2_times[frequency_count], trig_length)
                
                
                ### make SSVEP
                
                period = int(np.round(sample_rate/frequency))
                
                SSVEP = functions.make_SSVEPs(data_linear_interpolation, triggers, period)
                
               # plt.plot(SSVEP)
                
                # save amplitude
                SSVEP_amplitudes[subject-1,frequency_count,electrode,condition] = np.ptp(SSVEP)
                
                if condition == 0:
                    standing_SSVEP = np.copy(SSVEP)
                elif condition== 1:
                    walking_SSVEP = np.copy(SSVEP)
                    
            # save correlations and phase shift
            SSVEP_walking_standing_correlations[subject-1,frequency_count,electrode] = np.corrcoef(standing_SSVEP, walking_SSVEP)[0,1]
               
            SSVEP_phase_scores[subject-1,frequency_count,electrode] = functions.cross_correlation(standing_SSVEP, walking_SSVEP)

            frequency_count += 1
                    
                
######  plots

electrode = 0



for frequency_count in range(0,6):
    
    average_amplitude_O2 = statistics.median(SSVEP_amplitudes[:,frequency_count,electrode,0])
    average_SIGI_amplitude_O2 =  statistics.median(SIGI_amplitudes[:,frequency_count,electrode,0])
           
    print('  ')
    print(electrode_names[electrode] + '  ' + str(frequencies_to_use[frequency_count]) + ' Hz')
    print(' ' )
    print('Median SSVEP amplitude ' + str(frequencies_to_use[frequency_count]) + ' Hz ' + electrode_names[electrode] + ' = ' + str(average_amplitude_O2))
    print('Median SIGI amplitude  ' + str(frequencies_to_use[frequency_count])  + ' Hz ' + electrode_names[electrode] + '= ' + str(average_SIGI_amplitude_O2))

    
    
    
    plt.figure()
    plt.suptitle(str(frequencies_to_use[frequency_count]) + ' Hz')
    
    correlation_values = SSVEP_walking_standing_correlations[:,frequency_count, electrode]
    
    correlation_values_SIGI = SIGI_walking_standing_correlations[:,frequency_count,electrode]
    
    phase_shift_values = SSVEP_phase_scores[:,frequency_count, electrode]
    
    phase_shift_values_SIGI = SIGI_phase_scores[:,frequency_count,electrode]
    
    ###############  permutation tests  ####################
    
    
    condition_1_values =  correlation_values  #phase_shift_values ##
    condition_2_values = correlation_values_SIGI  ##phase_shift_values_SIGI  #
    
    true_difference =  (condition_1_values - condition_2_values).mean()
    
    num_loops = 1000
    
    shuffled_differences = np.zeros([num_loops,]) # empty array to put the shuffled differences into
    
    for loop in range(0,num_loops):
        
        # two temporary arrays, to put the shuffled values into
        temp_condition_1 = np.zeros([num_subjects,]) 
        temp_condition_2 = np.zeros([num_subjects,])
        
        for subject in range(0,num_subjects): # loop through each subject
    
            decide = choice(['yes', 'no'])  # for each subject, decide to either keep the correct labels, or swap the conditions. 50% chance
            
            if decide == 'yes': # keep the correct labels
                
                temp_condition_1[subject] = condition_1_values[subject] 
                temp_condition_2[subject] = condition_2_values[subject]
        
            elif decide == 'no': #swap the conditions
    
                temp_condition_1[subject] = condition_2_values[subject] 
                temp_condition_2[subject] = condition_1_values[subject]
    
    
        shuffled_differences[loop] = temp_condition_1.mean() - temp_condition_2.mean() # average the two shuffled conditions
        
        
    # plot histogram of the permutation test
        
    
    plt.subplot(1,2,1)
    
    plt.hist(shuffled_differences,10)
        
    plt.axvline(x=true_difference, color='r', linestyle='--')    
    
    
    
    Z_score = (true_difference - shuffled_differences.mean()) / np.std(shuffled_differences) # calculate Z score
    
    plt.title('Z score = '  + str(np.round(Z_score,2)))

    p_value_one_sided = scipy.stats.norm.sf(abs(Z_score)) #one-sided
    
    p_value_two_sided = scipy.stats.norm.sf(abs(Z_score))*2 #twosided
    
   
    print('Z score = ' + str(Z_score))
    print('p = ' + str(p_value_two_sided))
    
    
     # plot the average, std error and individual value
    plt.subplot(1,2,2) 
    
    average_condition_1 = condition_1_values.mean() # the average of the values from the first condition
    average_condition_2 = condition_2_values.mean() # the average of the values from the second condition
    
     # get the standard deviation
    std_deviation_1 = np.std(condition_1_values, axis = 0)
    std_deviation_2 = np.std(condition_2_values, axis = 0)
    
    # calculate standard error
    std_error_1 = std_deviation_1 / math.sqrt(10)
    std_error_2 = std_deviation_2 / math.sqrt(10)
    
    
    plt.scatter(np.zeros(10) + 1 , condition_1_values, color = 'b', label = 'standing vs standing SSVEP')
    plt.scatter(np.zeros(10) + 2 , condition_2_values, color = 'r', label = 'standing vs walking Signal generator')
    
    
    plt.errorbar(1, average_condition_1,yerr = std_error_1, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'g', ecolor='g')  
    plt.errorbar(2, average_condition_2,yerr = std_error_2, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'g', ecolor='g')  
    
    plt.xlim(0, 3)
    plt.ylim(-1, 1)

