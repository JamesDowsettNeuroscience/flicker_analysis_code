#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 12:35:34 2022

@author: James Dowsett
"""

from flicker_analysis_package import functions

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats

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






file_names = ('simulated_data_40Hz_Consistent_Amplitude', 'simulated_data_40Hz_Consistent_smaller_Amplitude', 'simulated_data_40Hz_Consistent_Amplitude_phase_shift')

## load simulated data and triggers

data_1 = np.load(file_names[0] + '.npy')

triggers_1 = np.load('simulated_triggers_40Hz.npy')

data_2 = np.load(file_names[2] + '.npy')

triggers_2 = np.load('simulated_triggers_40Hz.npy')


## compare two SSVEPs

num_subjects = 10

condition_1_values = np.zeros([num_subjects,])
condition_2_values = np.zeros([num_subjects,])

for subject in range(0,num_subjects):
    
    # plt.figure()
    # plt.title(subject)
    
    for condition in range(0,2):
    
        if condition == 0:
            data = np.copy(data_1)
            triggers = np.copy(triggers_1)
        elif condition == 1:
            data = np.copy(data_2)
            triggers = np.copy(triggers_2)
    
        SSVEPs = functions.compare_SSVEPs_split(data, triggers, 125)
        
        if condition == 0:
            SSVEP_1 = SSVEPs[0]
            SSVEP_2 = SSVEPs[1]
        elif condition == 1:  
            SSVEP_3 = SSVEPs[0]
            SSVEP_4 = SSVEPs[1]           
     
        
        # plt.plot(SSVEP_1, label = 'SSVEP_1')
        # plt.plot(SSVEP_2, label = 'SSVEP_2')
        # # plt.plot(SSVEP_3, label = 'SSVEP_3')
        # # plt.plot(SSVEP_4), label = 'SSVEP_4')
        # plt.plot(SSVEP_5, label = 'SSVEP_5')
        
       # plt.legend()
        
        # condition_1_values[subject] = np.ptp(SSVEP_1) - np.ptp(SSVEP_2)
        # condition_2_values[subject] = np.ptp(SSVEP_1) - np.ptp(SSVEP_5)
    phase_shift_1 = functions.cross_correlation(SSVEP_1, SSVEP_2)
    phase_shift_2 = functions.cross_correlation(SSVEP_1, SSVEP_3)

    diff_1 = np.ptp(SSVEP_1) - np.ptp(SSVEP_2) 
    diff_2 = np.ptp(SSVEP_1) - np.ptp(SSVEP_3) 

    condition_1_values[subject] = phase_shift_1
    condition_2_values[subject] = phase_shift_2
        
        # plt.title(str(np.round(phase_shift_1,2)) + '    ' + str(np.round(phase_shift_2,2)))
   
    
   
    
## permutation tests


average_condition_1 = condition_1_values.mean() # the average of the values from the first condition

average_condition_2 = condition_2_values.mean() # the average of the values from the second condition

# get the standard deviation
std_deviation_1 = np.std(condition_1_values, axis = 0)
std_deviation_2 = np.std(condition_2_values, axis = 0)

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
            
            temp_condition_1[subject] = condition_1_values[subject] # keep the correct labels
            temp_condition_2[subject] = condition_2_values[subject]
    
        elif decide == 'no':

            temp_condition_1[subject] = condition_2_values[subject] #swap the conditions
            temp_condition_2[subject] = condition_1_values[subject]


    average_shuffled_differences[loop] = temp_condition_1.mean() - temp_condition_2.mean() # average the two shuffled conditions
    
    
# import matplotlib.pyplot as plt
    
plt.figure()

plt.subplot(1,2,1)

plt.hist(average_shuffled_differences,10)
    
plt.axvline(x=true_difference, color='r', linestyle='--')    



Z_score = (true_difference - average_shuffled_differences.mean()) / np.std(average_shuffled_differences) # calculate Z score

plt.title('Z score = '  + str(np.round(Z_score,2)))

print('Z score = ' + str(Z_score))

p_value_one_sided = scipy.stats.norm.sf(abs(Z_score)) #one-sided

p_value_two_sided = scipy.stats.norm.sf(abs(Z_score))*2 #twosided

print('p = ' + str(p_value_two_sided))


plt.subplot(1,2,2)

plt.title('p = ' + str(np.round(p_value_two_sided,4)))

plt.scatter(np.zeros(len(condition_1_values)) + 1 , condition_1_values)
plt.scatter(np.zeros(len(condition_2_values)) + 2 , condition_2_values)


# divide by the square root of the number of subjects to get the standard error
std_error_1 = std_deviation_1 / math.sqrt(num_subjects)
std_error_2 = std_deviation_2 / math.sqrt(num_subjects)


# plot mean value with error bars
plt.errorbar(1, average_condition_1,yerr = std_error_1, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'g', ecolor='g', label='Insert Condition label here')  
plt.errorbar(2, average_condition_2,yerr = std_error_2, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'g', ecolor='g', label='Insert Condition label here')  


plt.xlim(0, 3)





# ## evoked fft test

# sample_rate = 5000

# length = 1

# evoked_fft = functions.evoked_fft(data_1, triggers_1, length, sample_rate)

# induced_fft = functions.induced_fft(data_1, triggers_1, length, sample_rate)

# length_of_segment = length * sample_rate
    
# time_vector = np.linspace(0, sample_rate, num=int(length_of_segment))
    
    
# plt.figure()

# plt.plot(time_vector,evoked_fft, label = 'Evoked')
# plt.plot(time_vector,induced_fft, label = 'Induced')

# plt.legend()

# plt.xlim([0, 100])




