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
trig_length = 4

#######################################


### Matrices to store results and SSVEPs

SIGI_SSVEPs = np.zeros([num_subjects,6,8,2,25]) # subject, frequency, electrode, condition, SSVEP data (SIGI always 40 Hz, lenght = 25)

SIGI_amplitudes = np.zeros([num_subjects,6,8,2]) # subject, frequency, electrode, condition

SIGI_walking_standing_correlations = np.zeros([num_subjects,6,8])

SIGI_phase_scores = np.zeros([num_subjects,6,8])


all_SSVEPs = np.zeros([num_subjects,6,8,2,34]) # subject, frequency, electrode, condition, SSVEP data (34 data points is the largest SSVEP)

SSVEP_amplitudes = np.zeros([num_subjects,6,8,2]) # subject, frequency, electrode , condition

SSVEP_walking_standing_correlations = np.zeros([num_subjects,6,8]) # subject, frequency, electrode

SSVEP_phase_scores = np.zeros([num_subjects,6,8])


blackout_SSVEPs = np.zeros([num_subjects,8,25]) # blackout was 40 Hz, so length = 25

blackout_amplitudes = np.zeros([num_subjects,8])

##################################

for subject in range(1,11):
   
    print('  ')
    print('Subject ' + str(subject))
    print(' ')
 
    
    for electrode in range(0,8):
        
        electrode_name = electrode_names[electrode]
        
        print(' ')
        print(electrode_name)
        print(' ')
        
        ## load raw data
        
        data_file_name = 'subject_' + str(subject) + '_electrode_' + str(electrode) + '_data.npy'
        
        raw_data = np.load(path + data_file_name)                



        ####### SIGI conditions
        
        frequency_count = 0
        for frequency in frequencies_to_use: # loop for each frequency to match the number of segments from each frequency 

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
                
                SSVEP = functions.make_SSVEPs(raw_data, triggers, 25) # SIGI was always 40 Hz, length = 25

                # plt.plot(SSVEP)
        
                SIGI_amplitudes[subject-1,frequency_count,electrode,condition] = np.ptp(SSVEP) # save amplitude
                
                SIGI_SSVEPs[subject-1,frequency_count,electrode,condition,:] = SSVEP # save the SSVEP
                
                    # make a copy to later compare walking and standing
                if condition == 0:
                    standing_SSVEP = np.copy(SSVEP)
                elif condition== 1:
                    walking_SSVEP = np.copy(SSVEP)
        
        
            # get walking/standing correlations and phase shift
            SIGI_walking_standing_correlations[subject-1,frequency_count,electrode] = np.corrcoef(standing_SSVEP, walking_SSVEP)[0,1]
        
            SIGI_phase_scores[subject-1,frequency_count,electrode] = functions.cross_correlation(standing_SSVEP, walking_SSVEP)
        
            frequency_count += 1
            
            
    
        ######### make real SSVEPs  ########################
        frequency_count = 0
        for frequency in frequencies_to_use:
            for condition in range(0,2):
            
            
                ## load triggers
                triggers_file_name = 'subject_' + str(subject) + '_' + condition_names[condition] + '_' + str(frequency) + 'Hz_triggers.npy'
                
                triggers = np.load(path + triggers_file_name)
                
                ### linear interpolation
                data_linear_interpolation = functions.linear_interpolation(raw_data, triggers, trig_1_times[frequency_count], trig_2_times[frequency_count], trig_length)
                
                
                
                ### make SSVEP
                
                period = int(np.round(sample_rate/frequency))
                
                SSVEP = functions.make_SSVEPs(data_linear_interpolation, triggers, period)

                # save amplitude
                SSVEP_amplitudes[subject-1,frequency_count,electrode,condition] = np.ptp(SSVEP)
                
                all_SSVEPs[subject-1,frequency_count,electrode,condition,0:len(SSVEP)] = SSVEP # save the SSVEP
                
                # make a copy to later compare walking and standing
                if condition == 0:
                    standing_SSVEP = np.copy(SSVEP)
                elif condition== 1:
                    walking_SSVEP = np.copy(SSVEP)
                    
            # save correlations and phase shift
            SSVEP_walking_standing_correlations[subject-1,frequency_count,electrode] = np.corrcoef(standing_SSVEP, walking_SSVEP)[0,1]
               
            SSVEP_phase_scores[subject-1,frequency_count,electrode] = functions.cross_correlation(standing_SSVEP, walking_SSVEP)

            frequency_count += 1
                    
                
    ############# make blackout SSVEPs  ######################
    
   
        
        ## load triggers
        triggers_file_name = 'subject_' + str(subject) + '_blackout_triggers.npy'
        
        triggers = np.load(path + triggers_file_name)
        
        ### linear interpolation, use 40 Hz trigger times, = frequency 2
        data_linear_interpolation = functions.linear_interpolation(raw_data, triggers, trig_1_times[2], trig_2_times[2], trig_length)
        
         
        period = int(np.round(sample_rate/40))
        
        SSVEP = functions.make_SSVEPs(data_linear_interpolation, triggers, period)
        
        blackout_amplitudes[subject-1,electrode] = np.ptp(SSVEP)

        blackout_SSVEPs[subject-1,electrode,:] = SSVEP # save the SSVEP



######  plots

## check raw SSVEPs for each electrode

electrode = 1 #('VEOG', 'blink', 'P3', 'P4', 'O2', 'Pz', 'O1', 'HEOG', 'x_dir', 'y_dir', 'z_dir')

for subject in range(1,11):
    
    plt.figure()
    plt.suptitle('Subject ' + str(subject) + ' ' + electrode_names[electrode])
    
    for frequency_count in range(0,6):
        
        plt.subplot(3,3,frequency_count+1)
        
        plt.title(str(frequencies_to_use[frequency_count]) + ' Hz')
        
        period = int(np.round(sample_rate/frequencies_to_use[frequency_count]))
        
        standing_SSVEP = all_SSVEPs[subject-1,frequency_count,electrode,0,0:period]
        walking_SSVEP = all_SSVEPs[subject-1,frequency_count,electrode,1,0:period]
        
        plt.plot(standing_SSVEP,'b')
        plt.plot(walking_SSVEP,'r')

    
    plt.subplot(3,3,3)

    blackout_SSVEP =  blackout_SSVEPs[subject-1,electrode,:] 
    
    plt.plot(blackout_SSVEP,'k')


    plt.subplot(3,3,8)
    for frequency_count in range(0,6):
        
        standing_SIGI = SIGI_SSVEPs[subject-1,frequency_count,electrode,0,:]
        walking_SIGI = SIGI_SSVEPs[subject-1,frequency_count,electrode,1,:]

        plt.plot(standing_SIGI,'b')
        plt.plot(walking_SIGI,'r')




###########  plot amplitudes

plt.figure()
plt.suptitle('All Amplitudes')

small_dot_size = 2

for electrode in range(0,8):
    
    for frequency_count in range(0,6):

        # standing
        all_subjects_amplitudes_standing = SSVEP_amplitudes[:,frequency_count,electrode,0]

        plt.scatter(np.zeros([10,])+((electrode*10) + frequency_count),all_subjects_amplitudes_standing, c='b', s=small_dot_size)

        mean_amplitude_standing = all_subjects_amplitudes_standing.mean()
        
        std_error_amplitude_standing = np.std(all_subjects_amplitudes_standing) / math.sqrt(10)

        plt.errorbar((electrode*10) + frequency_count, mean_amplitude_standing,yerr = std_error_amplitude_standing, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'b', ecolor='b')  

        # walking
        all_subjects_amplitudes_walking = SSVEP_amplitudes[:,frequency_count,electrode,1]
    
        plt.scatter(np.zeros([10,])+((electrode*10) + frequency_count+0.5),all_subjects_amplitudes_walking, c='r', s=small_dot_size)

        mean_amplitude_walking = all_subjects_amplitudes_walking.mean()
        
        std_error_amplitude_walking = np.std(all_subjects_amplitudes_walking) / math.sqrt(10)
        
        plt.errorbar((electrode*10) + frequency_count + 0.5, mean_amplitude_walking,yerr = std_error_amplitude_walking, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'r', ecolor='r')  
        
     #SIGI
    SIGI_amplitudes_standing = SIGI_amplitudes[:,frequency_count,electrode,0]  
    
    plt.scatter(np.zeros([10,])+((electrode*10) + 6),SIGI_amplitudes_standing, c='c', s=small_dot_size)
      
    mean_SIGI_standing = SIGI_amplitudes_standing.mean()
    
    std_error_SIGI_standing = np.std(SIGI_amplitudes_standing) / math.sqrt(10)
    
    plt.errorbar((electrode*10) + 6, mean_SIGI_standing,yerr = std_error_SIGI_standing, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'c', ecolor='c')  

    SIGI_amplitudes_walking = SIGI_amplitudes[:,frequency_count,electrode,1]  
    
    plt.scatter(np.zeros([10,])+((electrode*10) + 6.5),SIGI_amplitudes_walking, c='m', s=small_dot_size)
      
    mean_SIGI_walking = SIGI_amplitudes_walking.mean()
    
    std_error_SIGI_walking = np.std(SIGI_amplitudes_walking) / math.sqrt(10)
    
    plt.errorbar((electrode*10) + 6.5, mean_SIGI_walking,yerr = std_error_SIGI_walking, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'm', ecolor='m')  
  


    # blackout
    all_subjects_blackout =  blackout_amplitudes[:,electrode]   

    plt.scatter(np.zeros([10,])+(electrode*10) + 7,all_subjects_blackout, c='k', s=small_dot_size)

    mean_amplitude_blackout = np.nanmean(all_subjects_blackout)

    std_error_blackout = np.nanstd(all_subjects_blackout) / math.sqrt(10)
    
    plt.errorbar((electrode*10) + 7, mean_amplitude_blackout,yerr = std_error_blackout, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'k', ecolor='k')  
    
    
## set x axis ticks
x = np.arange(0,80,10)+3
y = np.zeros([8,])
labels = electrode_names[0:8]
plt.xticks(x, labels, rotation='vertical')
plt.show()




####### plot amplitude differences


plt.figure()
plt.suptitle('All Amplitude differences')

small_dot_size = 2

colours = ['r','m','g','b','c','y','k']

for electrode in range(0,8):
    
    for frequency_count in range(0,6):

        # standing
        all_subjects_amplitudes_standing = SSVEP_amplitudes[:,frequency_count,electrode,0]

          # walking
        all_subjects_amplitudes_walking = SSVEP_amplitudes[:,frequency_count,electrode,1]
        
        # difference
        amplitude_difference = all_subjects_amplitudes_walking - all_subjects_amplitudes_standing

        plt.scatter(np.zeros([10,])+((electrode*10) + frequency_count),amplitude_difference, c=colours[frequency_count], s=small_dot_size)

        mean_amplitude_difference = amplitude_difference.mean()
        
        std_error_amplitude_difference = np.std(amplitude_difference) / math.sqrt(10)

        plt.errorbar((electrode*10) + frequency_count, mean_amplitude_difference,yerr = std_error_amplitude_difference, solid_capstyle='projecting', capsize=5,  fmt='o', color= colours[frequency_count], ecolor=colours[frequency_count])  

    ## SIGI
    SIGI_amplitudes_standing = SIGI_amplitudes[:,frequency_count,electrode,0]
    SIGI_amplitudes_walking = SIGI_amplitudes[:,frequency_count,electrode,1]
  
    SIGI_difference = SIGI_amplitudes_walking - SIGI_amplitudes_standing
  
    plt.scatter(np.zeros([10,])+((electrode*10) + 6),SIGI_difference, c=colours[6], s=small_dot_size)

    mean_SIGI_difference = SIGI_difference.mean()
        
    std_error_SIGI_difference = np.std(SIGI_difference) / math.sqrt(10)
        
    plt.errorbar((electrode*10) + 6, mean_SIGI_difference,yerr = std_error_SIGI_difference, solid_capstyle='projecting', capsize=5,  fmt='o', color= colours[6], ecolor=colours[6])  

    
zero_line = np.arange(0,80)
plt.plot(zero_line,np.zeros([80,]),'k--')

## set x axis ticks
x = np.arange(0,80,10)+3
y = np.zeros([8,])
labels = electrode_names[0:8]
plt.xticks(x, labels, rotation='vertical')
plt.show()



##### plot phase shifts



plt.figure()
plt.suptitle('All Phase shifts')

small_dot_size = 2

colours = ['r','m','g','b','c','y','k']

for electrode in range(0,8):
    
    for frequency_count in range(0,6):

        # difference
        phase_scores_all_subjects =  SSVEP_phase_scores[:,frequency_count,electrode]

        plt.scatter(np.zeros([10,])+((electrode*10) + frequency_count),phase_scores_all_subjects, c=colours[frequency_count], s=small_dot_size)

        mean_phase_difference = phase_scores_all_subjects.mean()
        
        std_error_phase_difference = np.std(phase_scores_all_subjects) / math.sqrt(10)

        plt.errorbar((electrode*10) + frequency_count, mean_phase_difference,yerr = std_error_phase_difference, solid_capstyle='projecting', capsize=5,  fmt='o', color= colours[frequency_count], ecolor=colours[frequency_count])  

    ## SIGI

    SIGI_phase_difference_all_subjects = SIGI_phase_scores[:,frequency_count,electrode]
  
    plt.scatter(np.zeros([10,])+((electrode*10) + 6),SIGI_phase_difference_all_subjects, c=colours[6], s=small_dot_size)

    mean_SIGI_phase_difference = SIGI_phase_difference_all_subjects.mean()
        
    std_error_phase_SIGI_difference = np.std(SIGI_phase_difference_all_subjects) / math.sqrt(10)
        
    plt.errorbar((electrode*10) + 6, mean_SIGI_phase_difference,yerr = std_error_phase_SIGI_difference, solid_capstyle='projecting', capsize=5,  fmt='o', color= colours[6], ecolor=colours[6])  

    
# zero_line = np.arange(0,80)
# plt.plot(zero_line,np.zeros([80,]),'k--')

## set x axis ticks
x = np.arange(0,80,10)+3
y = np.zeros([8,])
labels = electrode_names[0:8]
plt.xticks(x, labels, rotation='vertical')
plt.show()




#### all correlations




plt.figure()
plt.suptitle('All correlations')

small_dot_size = 2

colours = ['r','m','g','b','c','y','k']

for electrode in range(0,8):
    
    for frequency_count in range(0,6):

        # difference
        correlations_all_subjects =  SSVEP_walking_standing_correlations[:,frequency_count,electrode]

        plt.scatter(np.zeros([10,])+((electrode*10) + frequency_count),correlations_all_subjects, c=colours[frequency_count], s=small_dot_size)

        mean_correlation = correlations_all_subjects.mean()
        
        std_error_correlations = np.std(correlations_all_subjects) / math.sqrt(10)

        plt.errorbar((electrode*10) + frequency_count, mean_correlation,yerr = std_error_correlations, solid_capstyle='projecting', capsize=5,  fmt='o', color= colours[frequency_count], ecolor=colours[frequency_count])  

    ## SIGI

    SIGI_correlations_all_subjects = SIGI_walking_standing_correlations[:,frequency_count,electrode]
  
    plt.scatter(np.zeros([10,])+((electrode*10) + 6),SIGI_correlations_all_subjects, c=colours[6], s=small_dot_size)

    mean_SIGI_correlation = SIGI_correlations_all_subjects.mean()
        
    std_error_SIGI_correlations = np.std(SIGI_correlations_all_subjects) / math.sqrt(10)
        
    plt.errorbar((electrode*10) + 6, mean_SIGI_correlation,yerr = std_error_SIGI_correlations, solid_capstyle='projecting', capsize=5,  fmt='o', color= colours[6], ecolor=colours[6])  

    
# zero_line = np.arange(0,80)
# plt.plot(zero_line,np.zeros([80,]),'k--')

## set x axis ticks
x = np.arange(0,80,10)+3
y = np.zeros([8,])
labels = electrode_names[0:8]
plt.xticks(x, labels, rotation='vertical')
plt.show()








# electrode = 3



# for frequency_count in range(0,6):
    
#     average_amplitude_O2 = statistics.median(SSVEP_amplitudes[:,frequency_count,electrode,0])
#     average_SIGI_amplitude_O2 =  statistics.median(SIGI_amplitudes[:,frequency_count,electrode,0])
           
#     print('  ')
#     print(electrode_names[electrode] + '  ' + str(frequencies_to_use[frequency_count]) + ' Hz')
#     print(' ' )
#     print('Median SSVEP amplitude ' + str(frequencies_to_use[frequency_count]) + ' Hz ' + electrode_names[electrode] + ' = ' + str(average_amplitude_O2))
#     print('Median SIGI amplitude  ' + str(frequencies_to_use[frequency_count])  + ' Hz ' + electrode_names[electrode] + '= ' + str(average_SIGI_amplitude_O2))

    
    
    
#     plt.figure()
#     plt.suptitle(electrode_names[electrode] + '  ' +  str(frequencies_to_use[frequency_count]) + ' Hz')
    
#     correlation_values = SSVEP_walking_standing_correlations[:,frequency_count, electrode]
    
#     correlation_values_SIGI = SIGI_walking_standing_correlations[:,frequency_count,electrode]
    
#     phase_shift_values = SSVEP_phase_scores[:,frequency_count, electrode]
    
#     phase_shift_values_SIGI = SIGI_phase_scores[:,frequency_count,electrode]
    
#     ###############  permutation tests  ####################
    
    
#     condition_1_values =  correlation_values  #phase_shift_values ##
#     condition_2_values = correlation_values_SIGI  ##phase_shift_values_SIGI  #
    
#     true_difference =  (condition_1_values - condition_2_values).mean()
    
#     num_loops = 1000
    
#     shuffled_differences = np.zeros([num_loops,]) # empty array to put the shuffled differences into
    
#     for loop in range(0,num_loops):
        
#         # two temporary arrays, to put the shuffled values into
#         temp_condition_1 = np.zeros([num_subjects,]) 
#         temp_condition_2 = np.zeros([num_subjects,])
        
#         for subject in range(0,num_subjects): # loop through each subject
    
#             decide = choice(['yes', 'no'])  # for each subject, decide to either keep the correct labels, or swap the conditions. 50% chance
            
#             if decide == 'yes': # keep the correct labels
                
#                 temp_condition_1[subject] = condition_1_values[subject] 
#                 temp_condition_2[subject] = condition_2_values[subject]
        
#             elif decide == 'no': #swap the conditions
    
#                 temp_condition_1[subject] = condition_2_values[subject] 
#                 temp_condition_2[subject] = condition_1_values[subject]
    
    
#         shuffled_differences[loop] = temp_condition_1.mean() - temp_condition_2.mean() # average the two shuffled conditions
        
        
#     # plot histogram of the permutation test
        
    
#     plt.subplot(1,2,1)
    
#     plt.hist(shuffled_differences,10)
        
#     plt.axvline(x=true_difference, color='r', linestyle='--')    
    
    
    
#     Z_score = (true_difference - shuffled_differences.mean()) / np.std(shuffled_differences) # calculate Z score
    
#     plt.title('Z score = '  + str(np.round(Z_score,2)))

#     p_value_one_sided = scipy.stats.norm.sf(abs(Z_score)) #one-sided
    
#     p_value_two_sided = scipy.stats.norm.sf(abs(Z_score))*2 #twosided
    
   
#     print('Z score = ' + str(Z_score))
#     print('p = ' + str(p_value_two_sided))
    
    
#      # plot the average, std error and individual values
#     plt.subplot(1,2,2) 
    
#     average_condition_1 = condition_1_values.mean() # the average of the values from the first condition
#     average_condition_2 = condition_2_values.mean() # the average of the values from the second condition
    
#      # get the standard deviation
#     std_deviation_1 = np.std(condition_1_values, axis = 0)
#     std_deviation_2 = np.std(condition_2_values, axis = 0)
    
#     # calculate standard error
#     std_error_1 = std_deviation_1 / math.sqrt(10)
#     std_error_2 = std_deviation_2 / math.sqrt(10)
    
    
#     plt.scatter(np.zeros(10) + 1 , condition_1_values, color = 'b', label = 'standing vs standing SSVEP')
#     plt.scatter(np.zeros(10) + 2 , condition_2_values, color = 'r', label = 'standing vs walking Signal generator')
    
    
#     plt.errorbar(1, average_condition_1,yerr = std_error_1, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'g', ecolor='g')  
#     plt.errorbar(2, average_condition_2,yerr = std_error_2, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'g', ecolor='g')  
    
#     plt.xlim(0, 3)
#     plt.ylim(-1, 1)

