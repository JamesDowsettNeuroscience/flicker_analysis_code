#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 15:04:55 2023

@author: James Dowsett
"""


### alpha optic flow part 2 analysis pipeline ######


from flicker_analysis_package import functions

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import math

import random

### information about the experiment:

#path = '/media/james/Expansion/alpha_optic_flow/experiment_2_data/'
path = '/home/james/Active_projects/alpha_optic_flow/experiment_2_data/'

electrode_names = ('O1', 'O2', 'P3', 'P4', 'P7', 'P8', 'REF2', 'EOG')

condition_names = ('Optic flow', 'Random', 'Static')

sample_rate = 1000

num_subjects = 20

length = 1 # length of FFT in seconds

# trigger times for linear interpolation
trigger_1_times = [-1, -1]
trigger_2_times = [54, 45]
trig_length = 6


subjects_to_use = [ 1,  2,  3, 5,  6,  7,  8,  9, 10, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

decoding_accuracy = np.zeros([len(subjects_to_use),8,2])
subject_count = 0

for subject in subjects_to_use:
    
    # plt.figure()
    # plt.suptitle('Subject ' + str(subject))
    
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
            
          #  electrode = 0
     
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
             
             
            max_data_length = max(len(data_1), len(data_2), len(data_3))
            min_number_of_triggers = min(len(triggers_1),len(triggers_2),len(triggers_3))
            
            # num_triggers = 1000
            # if min_number_of_triggers < num_triggers:
            #     num_triggers = min_number_of_triggers
          
            num_triggers = min_number_of_triggers    
           
           
            # first put all data  into one matrix
            data_all_conditions = np.zeros([3,max_data_length])
            data_all_conditions[0,0:len(data_1)] = data_1
            data_all_conditions[1,0:len(data_2)] = data_2
            data_all_conditions[2,0:len(data_3)] = data_3
               
             # put all triggers into one matrix
            triggers_all_conditions = np.zeros([3,num_triggers])
            triggers_all_conditions = triggers_all_conditions.astype(int)
            triggers_all_conditions[0,:] = triggers_1[0:num_triggers]
            triggers_all_conditions[1,:] = triggers_2[0:num_triggers]
            triggers_all_conditions[2,:] = triggers_3[0:num_triggers]
            
            
           
            num_loops = 10
             
            print('\nSubject ' + str(subject) + ' block' + str(block) + ' ' + electrode_names[electrode])
            #print('Number of triggers = ' + str(num_triggers))
            
            # Decode
            average_percent_correct = functions.decode_correlations(data_all_conditions, triggers_all_conditions, 3, num_triggers, period, num_loops)
             
            print('Decoding accuracy = ' + str(average_percent_correct) + '\n')
            
            decoding_accuracy[subject_count,electrode,block-1] = average_percent_correct
            
    
            
            
            # # make SSVEPs
            # SSVEP_1 = functions.make_SSVEPs(data_1, triggers_all_conditions[0,0:num_triggers], period)
            # SSVEP_2 = functions.make_SSVEPs(data_2, triggers_all_conditions[1,0:num_triggers], period)
            # SSVEP_3 = functions.make_SSVEPs(data_3, triggers_all_conditions[2,0:num_triggers], period)
            
            # if block == 1:
            #     plt.subplot(1,2,1)
            #     plt.title('9 Hz  ' + str(np.round(average_percent_correct)) + ' %')
            # elif block == 2:
            #     plt.subplot(1,2,2)
            #     plt.title('11 Hz  ' + str(np.round(average_percent_correct)) + ' %')
             
            
            # plt.plot(SSVEP_1, label = condition_names[0])
            # plt.plot(SSVEP_2, label = condition_names[1])
            # plt.plot(SSVEP_3, label = condition_names[2])
            
            # plt.legend()
        
        
        
    subject_count += 1
    
    
    
    
    
## plot grand average decoding scores

import statistics

plt.figure()

for electrode in range(0,8):
    
    scores_block_1 = decoding_accuracy[:,electrode,0]
    scores_block_2 = decoding_accuracy[:,electrode,1]
    
    mean_block_1 = scores_block_1.mean()
    mean_block_2 = scores_block_2.mean()
    
    median_block_1 = statistics.median(scores_block_1)
    median_block_2 = statistics.median(scores_block_2)
    
    
    std_error_block_1 = np.std(scores_block_1) / math.sqrt(len(scores_block_1))
    std_error_block_2 = np.std(scores_block_2) / math.sqrt(len(scores_block_2))


    plt.scatter(np.zeros([len(scores_block_1),])+electrode,scores_block_1,s=1, c='b')
    plt.scatter(np.zeros([len(scores_block_1),])+electrode+0.2,scores_block_2,s=1, c='r')
  

    plt.errorbar(electrode, mean_block_1,yerr = std_error_block_1, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'b', ecolor='b')  
    plt.errorbar(electrode+0.2, mean_block_2,yerr = std_error_block_2, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'r', ecolor='b')  

    
    plt.plot([electrode, electrode+0.2], [mean_block_1, mean_block_2], 'c', label = electrode_names[electrode])

plt.axhline(y = 33.33, color = 'k', linestyle = '--')

#plt.legend()
x = np.arange(0,8)
plt.xticks(x, electrode_names[0:8])    
  
plt.ylim([0,100])  
plt.ylabel('Average Decoding Accuracy %')