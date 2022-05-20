#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 12:19:03 2022

@author: James Dowsett
"""

from flicker_analysis_package import functions

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats
from random import choice

### information about the experiment: Gamma walk 1

path = '/home/james/Active_projects/Gamma_walk/Gamma_walking_experiment_1/raw_data_for_analysis_package/'

electrode_names = ('VEOG', 'blink', 'P3', 'P4', 'O2', 'Pz', 'O1', 'HEOG', 'x_dir', 'y_dir', 'z_dir')

condition_names = ('standing', 'walking')

sample_rate = 1000

num_subjects = 10

frequencies_to_use = (30, 35, 40, 45, 50, 55)

#######################################



electrode = 6

electrode_name = electrode_names[electrode]


## empty matrices to put output scores into
amplitude_scores = np.zeros([num_subjects, len(frequencies_to_use), len(condition_names)]) # matrix to put the amplitudes into

amplitude_scores_split = np.zeros([num_subjects, len(frequencies_to_use), len(condition_names), 2]) # scores of the random split data, 2 per conditions

phase_scores_split = np.zeros([num_subjects,len(frequencies_to_use),6])

correlation_scores_split = np.zeros([num_subjects,len(frequencies_to_use),6])


for subject in range(1,num_subjects+1):
    
    print(' ')
    print('Subject ' + str(subject))
    
    plt.figure()
    plt.suptitle(subject)
    
    
    plot_count = 1
    frequency_count = 0
    for frequency in frequencies_to_use:
        
        print(str(frequency) + ' Hz')
        plt.subplot(2,3,plot_count)
        plt.title(str(frequency) + ' Hz')

        # loop each condition multiple times to get a more stable value of the phase difference
        num_loops = 100
        num_combinations = 6 # the number of combinations for phase differences, for 2 conditions this is 4 SSVEPs which is 6 possible combinations
        
        phase_scores = np.zeros([num_combinations,num_loops])
        correlation_scores = np.zeros([num_combinations,num_loops])

        for loop in range(0,num_loops):

            for condition in range(0, len(condition_names)): # standing = 0, walking = 1
                
                condition_name = condition_names[condition]
                
                ## load data
                data_file_name = 'subject_' + str(subject) + '_electrode_' + str(electrode) + '_data.npy'
                
                data = np.load(path + data_file_name)                
                
            
                
                
                
                ## load triggers
                triggers_file_name = 'subject_' + str(subject) + '_' + condition_name + '_' + str(frequency) + 'Hz_triggers.npy'
                
                triggers = np.load(path + triggers_file_name)
                
                
                
                
                ### make SSVEP
                
                period = int(np.round(sample_rate/frequency))
                
                SSVEP = functions.make_SSVEPs(data, triggers, period)
    
              #  SNR = functions.SNR_random(data, triggers, period)
                
                # random split into two SSVEPs
                split_SSVEPs = functions.compare_SSVEPs_split(data, triggers, period)
                
                SSVEP_1 = split_SSVEPs[0]
                SSVEP_2 = split_SSVEPs[1]
                
                ## record scores
                amplitude_scores[subject-1, frequency_count, condition] = np.ptp(SSVEP)
                amplitude_scores_split[subject-1, frequency_count, condition,0] = np.ptp(SSVEP_1)
                amplitude_scores_split[subject-1, frequency_count, condition,1] = np.ptp(SSVEP_2)
                
                
                ## store the SSVEPs to compare them later
                if condition == 0: # standing
                    condition_1_split_SSVEP_1 = np.copy(SSVEP_1)
                    condition_1_split_SSVEP_2 = np.copy(SSVEP_2)
                elif condition ==  1: # walking
                    condition_2_split_SSVEP_1 = np.copy(SSVEP_1)
                    condition_2_split_SSVEP_2 = np.copy(SSVEP_2)
               
                ## plot  SSVEPs and one set of splits as an example  
                if loop == 0:
                    if condition == 0: # standing
                        plt.plot(SSVEP, 'b', label = condition_names[condition])#, label = (condition_name + ' ' + str(np.round(SNR,2))))
                        plt.plot(SSVEP_1,'c')
                        plt.plot(SSVEP_2,'c')                 
                    elif condition ==  1: # walking
                        plt.plot(SSVEP, 'r', label = condition_names[condition])#, label = (condition_name + ' ' + str(np.round(SNR,2))))
                        plt.plot(SSVEP_1,'m')
                        plt.plot(SSVEP_2,'m')
                        
           
                
               
            ## get phase shift scores for all 6 possible combinations (2 conditions = 4 50% splits)
        
            phase_scores[0,loop] = functions.cross_correlation(condition_1_split_SSVEP_1, condition_1_split_SSVEP_2)
            phase_scores[1,loop] = functions.cross_correlation(condition_1_split_SSVEP_1, condition_2_split_SSVEP_1)
            phase_scores[2,loop] = functions.cross_correlation(condition_1_split_SSVEP_1, condition_2_split_SSVEP_2)
            phase_scores[3,loop] = functions.cross_correlation(condition_1_split_SSVEP_2, condition_2_split_SSVEP_1)
            phase_scores[4,loop] = functions.cross_correlation(condition_1_split_SSVEP_2, condition_2_split_SSVEP_2)
            phase_scores[5,loop] = functions.cross_correlation(condition_2_split_SSVEP_1, condition_2_split_SSVEP_2)

            ## get correlation scores
            correlation_scores[0,loop] = np.corrcoef(condition_1_split_SSVEP_1, condition_1_split_SSVEP_2)[0,1]
            correlation_scores[1,loop] = np.corrcoef(condition_1_split_SSVEP_1, condition_2_split_SSVEP_1)[0,1]
            correlation_scores[2,loop] = np.corrcoef(condition_1_split_SSVEP_1, condition_2_split_SSVEP_2)[0,1]
            correlation_scores[3,loop] = np.corrcoef(condition_1_split_SSVEP_2, condition_2_split_SSVEP_1)[0,1]
            correlation_scores[4,loop] = np.corrcoef(condition_1_split_SSVEP_2, condition_2_split_SSVEP_2)[0,1]
            correlation_scores[5,loop] = np.corrcoef(condition_2_split_SSVEP_1, condition_2_split_SSVEP_2)[0,1]
                


        # average values from each loop and put into matrix
        for combination in range(0,6):
            phase_scores_split[subject-1, frequency_count,combination] = phase_scores[combination,:].mean(axis=0)# average the phase scores from all the loops
            correlation_scores_split[subject-1, frequency_count,combination] = correlation_scores[combination,:].mean(axis=0)# average the correlation scores
        

# put the scores into the title of the figure
        condition_1_split_correlation = correlation_scores_split[subject-1, frequency_count,0]
        condition_2_split_correlation = correlation_scores_split[subject-1, frequency_count,5]
        plt.title(str(frequency) + ' Hz  ' + str(np.round(condition_1_split_correlation,2)) + '  ' + str(np.round(condition_2_split_correlation,2)))
        
        # condition_1_split_phase_shift = phase_scores_split[subject-1, frequency_count,0]
        # condition_2_split_phase_shift = phase_scores_split[subject-1, frequency_count,5]        
        # plt.title(str(frequency) + ' Hz  ' + str(np.round(condition_1_split_phase_shift,2)) + '  ' + str(np.round(condition_2_split_phase_shift,2)))
        


        
        plt.legend()
        plot_count += 1
        frequency_count +=1  
        
      
        
### permutation tests

for frequency_count in range(0,len(frequencies_to_use)):
    
    
    ## reject subjects with low/variable signal 
    correlation_split_condition_1 = correlation_scores_split[:,frequency_count,0] # compare condition 1 with itself (split 1 vs 2), 
    correlation_split_condition_2 = correlation_scores_split[:,frequency_count,5] #condition 2 split 1 with condition 2 split 2   
    
    corr_cutoff = 0.33 # correlations above this value are significant 
    
    # subjects_to_use = []
    # for subject in range(0,num_subjects):
    #     if (correlation_split_condition_1[subject] > corr_cutoff) and (correlation_split_condition_2[subject] > corr_cutoff):
    #         subjects_to_use.append(subject)
    subjects_to_use = np.arange(0,10)

    # condition_1_values = amplitude_scores[:,frequency_count,0]
    # condition_2_values = amplitude_scores[:,frequency_count,1]
    
    # condition_1_values = phase_scores_split[:,frequency_count,0] # compare condition 1 with itself (split 1 vs 2), 
    # condition_2_values = phase_scores_split[:,frequency_count,1] #condition 1 split 1 with condition 2 split 1
    # condition_3_values = phase_scores_split[:,frequency_count,5] #condition 2 split 1 with condition 2 split 2


    condition_1_values = correlation_scores_split[:,frequency_count,0] # compare condition 1 with itself (split 1 vs 2), 
    condition_2_values = correlation_scores_split[:,frequency_count,1] #condition 1 split 1 with condition 2 split 1
    condition_3_values = correlation_scores_split[:,frequency_count,5] #condition 2 split 1 with condition 2 split 2
    
    

    
    
    true_difference = (condition_1_values[subjects_to_use] - condition_2_values[subjects_to_use]).mean()

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
    
    
        shuffled_differences[loop] = temp_condition_1[subjects_to_use].mean() - temp_condition_2[subjects_to_use].mean() # average the two shuffled conditions
        
        
    # plot histogram of the permutation test
        
    plt.figure()
    
    plt.subplot(1,2,1)
    
    plt.hist(shuffled_differences,10)
        
    plt.axvline(x=true_difference, color='r', linestyle='--')    
    
    
    
    Z_score = (true_difference - shuffled_differences.mean()) / np.std(shuffled_differences) # calculate Z score
    
    plt.title('Z score = '  + str(np.round(Z_score,2)))
    
    print('Z score = ' + str(Z_score))
    
    p_value_one_sided = scipy.stats.norm.sf(abs(Z_score)) #one-sided
    
    p_value_two_sided = scipy.stats.norm.sf(abs(Z_score))*2 #twosided
    
    print('p = ' + str(p_value_two_sided))
    
    
    
    # plot the average, std error and individual values
    plt.subplot(1,2,2) 
    
    average_condition_1 = condition_1_values[subjects_to_use].mean() # the average of the values from the first condition
    average_condition_2 = condition_2_values[subjects_to_use].mean() # the average of the values from the second condition
    
    average_condition_3 = condition_3_values[subjects_to_use].mean() 
    
    # get the standard deviation
    std_deviation_1 = np.std(condition_1_values[subjects_to_use], axis = 0)
    std_deviation_2 = np.std(condition_2_values[subjects_to_use], axis = 0)
    std_deviation_3 = np.std(condition_3_values[subjects_to_use], axis = 0)
    
    plt.title('p = ' + str(np.round(p_value_two_sided,4)))
    
    # plt.scatter(np.zeros(len(condition_1_values)) + 1 , condition_1_values, color = 'b', label = condition_names[0])
    # plt.scatter(np.zeros(len(condition_2_values)) + 2 , condition_2_values, color = 'r', label = condition_names[1])
    
    plt.scatter(np.zeros(len(subjects_to_use)) + 1 , condition_1_values[subjects_to_use], color = 'b', label = 'standing vs standing')
    plt.scatter(np.zeros(len(subjects_to_use)) + 2 , condition_2_values[subjects_to_use], color = 'r', label = 'standing vs walking')
    plt.scatter(np.zeros(len(subjects_to_use)) + 3 , condition_3_values[subjects_to_use], color = 'g', label = 'walking vs walking')
    
    
    # divide by the square root of the number of subjects to get the standard error
    std_error_1 = std_deviation_1 / math.sqrt(len(subjects_to_use))
    std_error_2 = std_deviation_2 / math.sqrt(len(subjects_to_use))
    std_error_3 = std_deviation_3 / math.sqrt(len(subjects_to_use))
    
    
    # plot mean value with error bars
    plt.errorbar(1, average_condition_1,yerr = std_error_1, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'g', ecolor='g')  
    plt.errorbar(2, average_condition_2,yerr = std_error_2, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'g', ecolor='g')  
    plt.errorbar(3, average_condition_3,yerr = std_error_3, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'g', ecolor='g')  
    
    plt.xlim(0, 4)


    plt.suptitle(str(frequencies_to_use[frequency_count]) + ' Hz  ' + str(len(subjects_to_use)) + ' good subjects')

    plt.legend()
