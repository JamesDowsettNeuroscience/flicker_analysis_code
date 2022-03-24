#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 12:35:34 2022

@author: James Dowsett
"""

from flicker_analysis_package import functions

import numpy as np
import matplotlib.pyplot as plt


# file_names = ('simulated_data_40Hz_Consistent_Amplitude', 'simulated_data_40Hz_Consistent_smaller_Amplitude')

# for condition in(0,1):
    
#     ## load simulated data and triggers
    
#     data = np.load(file_names[condition] + '.npy')
    
#     triggers = np.load('simulated_triggers_40Hz.npy')
    
    
#     #plt.plot(data) # plot before filter
    
#     ## filter
#     data = functions.high_pass_filter(5000, data)
    
    
#     #plt.plot(data) # plot after filter
    
    
#     ## make SSVEP
    
#     SSVEP = functions.make_SSVEPs(data, triggers, 125)
    
#     plt.plot(SSVEP)
    
#     num_loops = 100
    
#     # functions.make_SSVEPs_random(data, triggers, 125, num_loops)
   
   
   
   


# sample_rate = 1000

# t = np.arange(0,100)

# SSVEP_1 = np.sin(2 * np.pi * 1/sample_rate * 10 * t)

# SSVEP_2 = np.sin(2 * np.pi * 1/sample_rate * 10 * (t+30))

# plt.plot(SSVEP_1)

# plt.plot(SSVEP_2)

# phase_lag = functions.cross_correlation(SSVEP_1, SSVEP_2)


# phase_lag_degrees = phase_lag/len(SSVEP_1) * 360

# print('Phase lag in degrees = ' + str(phase_lag_degrees))






file_names = ('simulated_data_40Hz_Consistent_Amplitude', 'simulated_data_40Hz_Consistent_smaller_Amplitude')

## load simulated data and triggers

data_1 = np.load(file_names[0] + '.npy')

triggers_1 = np.load('simulated_triggers_40Hz.npy')

data_2 = np.load(file_names[1] + '.npy')

triggers_2 = np.load('simulated_triggers_40Hz.npy')


## compare two SSVEPs

num_subjects = 8

condition_1_differences = np.zeros([num_subjects,])
condition_2_differences = np.zeros([num_subjects,])

for subject in range(0,num_subjects):
    
    # plt.figure()
    # plt.title(subject)
    
    SSVEPs = functions.compare_SSVEPs_split(data_1, data_2, triggers_1, triggers_2, 125)
    
    SSVEP_1 = SSVEPs[0]
    SSVEP_2 = SSVEPs[1]
    SSVEP_3 = SSVEPs[2]
    SSVEP_4 = SSVEPs[3]
    
    SSVEP_5 = (SSVEP_3 + SSVEP_4) /2
    
    # plt.plot(SSVEP_1)
    # plt.plot(SSVEP_2)
    # plt.plot(SSVEP_3)
    # plt.plot(SSVEP_4)
    # plt.plot(SSVEP_5)
    
    
    
    condition_1_differences[subject] = np.ptp(SSVEP_1) - np.ptp(SSVEP_2)
    condition_2_differences[subject] = np.ptp(SSVEP_1) - np.ptp(SSVEP_5)


## permutation tests


average_condition_1 = condition_1_differences.mean() # the average of the values from the first condition

average_condition_2 = condition_2_differences.mean() # the average of the values from the second condition

true_difference = average_condition_1 - average_condition_2

from random import choice

num_loops = 1000

average_shuffled_differences = np.zeros([num_loops,]) # empty array to put the shuffled differences into

for loop in range(0,num_loops):
    
    # two temporary arrays, to put the shuffled values into
    temp_condition_1 = np.zeros([num_subjects,]) 
    temp_condition_2 = np.zeros([num_subjects,])
    
    for subject in range(0,num_subjects): # loop through each subject
        
        decide = choice(['yes', 'no'])  # for each subject, decide to either keep the correct labels, or swap the conditions. 50% chance
        
        if decide == 'yes':
            
            temp_condition_1[subject] = condition_1_differences[subject] # keep the correct labels
            temp_condition_2[subject] = condition_2_differences[subject]
    
        elif decide == 'no':

            temp_condition_1[subject] = condition_2_differences[subject] #swap the conditions
            temp_condition_2[subject] = condition_1_differences[subject]


    average_shuffled_differences[loop] = temp_condition_1.mean() - temp_condition_2.mean() # average the two shuffled conditions
    
    
# import matplotlib.pyplot as plt
    
plt.figure()

plt.hist(average_shuffled_differences,10)
    
plt.axvline(x=true_difference, color='r', linestyle='--')    

Z_score = (true_difference - average_shuffled_differences.mean()) / np.std(average_shuffled_differences) # calculate Z score

print('Z score = ' + str(Z_score))
