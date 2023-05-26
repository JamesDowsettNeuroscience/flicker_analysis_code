#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 15:20:59 2023

@author: James Dowsett
"""

## alpha optic flow: decoding accuracy with variable amount of triggers


from flicker_analysis_package import functions

import numpy as np
import matplotlib.pyplot as plt


### information about the experiment:

#path = '/media/james/Expansion/alpha_optic_flow/experiment_2_data/'
path = '/home/james/Active_projects/alpha_optic_flow/experiment_2_data/'

electrode_names = ('O1', 'O2', 'P3', 'P4', 'P7', 'P8', 'REF2', 'EOG')

condition_names = ('Optic flow', 'Random', 'Static')

sample_rate = 1000

num_subjects = 20


# trigger times for linear interpolation
trigger_1_times = [-1, -1]
trigger_2_times = [54, 45]
trig_length = 6


subjects_to_use = [1, 5,  6,  7,  8,  9, 10, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 25, 27, 29, 30, 31]

smallest_period = 90


numbers_of_triggers_to_use = np.arange(5,100,10)

decoding_accuracy = np.zeros([len(numbers_of_triggers_to_use),len(subjects_to_use),8,2])



num_triggers_count = 0

for num_triggers in numbers_of_triggers_to_use:
    
    subject_count = 0
    
    for subject in subjects_to_use:
        
        for block in (1,2): # 9 and 11 Hz flicker blocks
    
            if block == 1:
                period = 110
                time_1 = trigger_1_times[0]
                time_2 = trigger_2_times[0]    
            elif block == 2:
                period = 90
                time_1 = trigger_1_times[1]
                time_2 = trigger_2_times[1]
             
             
    
            for electrode in range(0,8):
                

                # load data and triggers condition 1
                file_name = 'S' + str(subject) + '_block' + str(block) + '_cond1'
        
                all_data = np.load(path + file_name + '_all_data.npy')
        
                data_1 = all_data[electrode,:]
                REF_2_data = all_data[6,:] # get the data for the second reference, to re-reference (left and right ears)
                data_1 = data_1 - (REF_2_data/2) # re-reference    
        
                triggers_1 = np.load(path + file_name + '_all_triggers.npy')
                data_1 = functions.linear_interpolation(data_1, triggers_1, time_1, time_2, trig_length) # linear interpolation
                
                
                # load data and triggers condition 2
                file_name = 'S' + str(subject) + '_block' + str(block) + '_cond2'
        
                all_data = np.load(path + file_name + '_all_data.npy')
        
                data_2 = all_data[electrode,:]
                REF_2_data = all_data[6,:] # get the data for the second reference, to re-reference (left and right ears)
                data_2 = data_2 - (REF_2_data/2) # re-reference    
        
                triggers_2 = np.load(path + file_name + '_all_triggers.npy')
                data_2 = functions.linear_interpolation(data_2, triggers_2, time_1, time_2, trig_length) # linear interpolation
                
                
                
                # load data and triggers condition 3
                file_name = 'S' + str(subject) + '_block' + str(block) + '_cond3'
        
                all_data = np.load(path + file_name + '_all_data.npy')
        
                data_3 = all_data[electrode,:]
                REF_2_data = all_data[6,:] # get the data for the second reference, to re-reference (left and right ears)
                data_3 = data_3 - (REF_2_data/2) # re-reference    
        
                triggers_3 = np.load(path + file_name + '_all_triggers.npy')
                data_3 = functions.linear_interpolation(data_3, triggers_3, time_1, time_2, trig_length) # linear interpolation
                
                            
                 ## decode with universal decoder function
                 
                 # find the longest data, so all data arrays fit into the matrix
                max_data_length = max(len(data_1), len(data_2), len(data_3))
                # find the condition with the minimum number of triggers, so all conditions can have the same number of triggers
                min_number_of_triggers = min(len(triggers_1),len(triggers_2),len(triggers_3))
                
                num_triggers_this_block = num_triggers
                
                if min_number_of_triggers < num_triggers:
                    num_triggers_this_block = min_number_of_triggers
                    print('Not enough triggers')
              
               
                # first put all data  into one matrix
                data_all_conditions = np.zeros([3,max_data_length])
                data_all_conditions[0,0:len(data_1)] = data_1
                data_all_conditions[1,0:len(data_2)] = data_2
                data_all_conditions[2,0:len(data_3)] = data_3
            
                
                 # put all triggers into one matrix
                start_trigger = 0
                 
                triggers_all_conditions = np.zeros([3,num_triggers_this_block])
                triggers_all_conditions = triggers_all_conditions.astype(int)
                triggers_all_conditions[0,:] = triggers_1[start_trigger:start_trigger+num_triggers_this_block]
                triggers_all_conditions[1,:] = triggers_2[start_trigger:start_trigger+num_triggers_this_block]
                triggers_all_conditions[2,:] = triggers_3[start_trigger:start_trigger+num_triggers_this_block]
                
                
               
                num_loops = 10
                 
                
                print('\n' + str(num_triggers) + ' triggers')
                print('Subject ' + str(subject) + ' block' + str(block) + ' ' + electrode_names[electrode])
                #print('Number of triggers = ' + str(num_triggers))
                
                # Decode, use smallest Period, so same amount of data is being used in both conditions
                #average_percent_correct = functions.decode_correlations(data_all_conditions, triggers_all_conditions, 3, num_triggers, smallest_period, num_loops)
               
                # Decode with full length of SSVEP
                average_percent_correct = functions.decode_correlations(data_all_conditions, triggers_all_conditions, 3, num_triggers_this_block, smallest_period, num_loops)
                  
                
                print('Decoding accuracy = ' + str(average_percent_correct) + '\n')
                
                decoding_accuracy[num_triggers_count,subject_count,electrode,block-1] = average_percent_correct
    
    
            
        subject_count += 1
        
        
    num_triggers_count += 1  
    
    
np.save(path + 'decoding_accuracy_by_num_triggers',decoding_accuracy)
    


#########  Plots

import math

plt.figure()


electrodes_to_use = [0,1,2,3,4,5,7]

electrode_names_to_use = ('O1', 'O2', 'P3', 'P4', 'P7', 'P8', 'EOG')


electrode_count = 0
for electrode in electrodes_to_use:
    
    plt.subplot(2,4,electrode_count+1)
    plt.title(electrode_names_to_use[electrode_count])

    block_1_all_scores = decoding_accuracy[:,:,electrode,0]
    block_2_all_scores = decoding_accuracy[:,:,electrode,1]
    
    average_scores_block_1 = block_1_all_scores.mean(axis=1)    
    average_scores_block_2 = block_2_all_scores.mean(axis=1)    
    
    plt.plot(numbers_of_triggers_to_use, average_scores_block_1, 'b', label = '9 Hz')
    plt.plot(numbers_of_triggers_to_use, average_scores_block_2, 'r', label = '11 Hz')
    
    
    plt.legend()
    plt.ylim([0, 100])
    
    plt.axhline(y = 33.33, color = 'k', linestyle = '--')
    
    if electrode_count == 0 or electrode_count == 4:
        plt.ylabel('Average Decoding accuracy %')
        
    plt.xlabel('Number of triggers')
    
    for trigger_amount in range(0,len(numbers_of_triggers_to_use)):
        
        block_1_scores = decoding_accuracy[trigger_amount,:,electrode,0]
        block_2_scores = decoding_accuracy[trigger_amount,:,electrode,1]
        
        mean_block_1 = block_1_scores.mean()
        mean_block_2 = block_2_scores.mean()
        
        std_error_block_1 = np.std(block_1_scores) / math.sqrt(len(block_1_scores))
        std_error_block_2 = np.std(block_2_scores) / math.sqrt(len(block_2_scores))
 
        plt.errorbar(numbers_of_triggers_to_use[trigger_amount], mean_block_1,yerr = std_error_block_1, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'b', ecolor='b')  
        plt.errorbar(numbers_of_triggers_to_use[trigger_amount], mean_block_2,yerr = std_error_block_2, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'r', ecolor='r')  

    
    electrode_count += 1