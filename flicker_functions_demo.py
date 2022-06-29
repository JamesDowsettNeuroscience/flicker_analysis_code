#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 14:13:01 2022

@author: James Dowsett
"""

#### flicker analysis functions demo

from flicker_analysis_package import functions

import numpy as np
import matplotlib.pyplot as plt


file_names = ('simulated_data_40Hz_Consistent_Amplitude', 'simulated_data_40Hz_Consistent_smaller_Amplitude', 'simulated_data_40Hz_Consistent_Amplitude_phase_shift')


frequency = 40

sample_rate = 5000

period = int(5000/frequency)


for condition in range(0,3):

    print(' ')
    print(' ')
    print(file_names[condition])    

    ## load simulated data and triggers
    
    data = np.load(file_names[condition] + '.npy')
    
    triggers = np.load('simulated_triggers_40Hz.npy')
    

    ### make SSVEPs
 
    SSVEP = functions.make_SSVEPs(data, triggers, period)
    
    ## plot SSVEPs
    plt.figure(1)
    plt.plot(SSVEP, label = file_names[condition])
    
    ## save SSVEPs for later
    if condition == 0:
        SSVEP_0 = np.copy(SSVEP)
    elif condition == 1:
        SSVEP_1 = np.copy(SSVEP)
    elif condition == 2:
        SSVEP_2 = np.copy(SSVEP)

    
    ## 50/50 split of the data to make two SSVEPs and compare
    split_correlation_value = functions.compare_SSVEPs_split(data, triggers, period)
    print(' ')
    print('50/50 split correlation value = ' + str(split_correlation_value))
    
    
    ###  signal to noise ratio by randomly shuffling the data points once
    # SNR_shuffle = functions.SNR_random(data, triggers, period)
    # print('Signal to Noise ratio shuffled once = ' + str(SNR_shuffle))
    
    ### calculate signal to noise ratio with permutation test on shuffled data
    num_loops = 10
    print(' ')
    Z_score = functions.make_SSVEPs_random(data, triggers, period, num_loops)
    



###### 


plt.figure(1)
plt.legend()



### compare SSVEPs phase shift
print(' ')

phase_shift_0_1 = functions.cross_correlation(SSVEP_0, SSVEP_1)
print('Phase shift between SSVEP 0 and SSVEP 1 = ' + str(phase_shift_0_1) + ' degrees')

phase_shift_0_2 = functions.cross_correlation(SSVEP_0, SSVEP_2)
print('Phase shift between SSVEP 0 and SSVEP 2 = ' + str(phase_shift_0_1) + ' degrees')

phase_shift_1_2 = functions.cross_correlation(SSVEP_1, SSVEP_2)
print('Phase shift between SSVEP 1 and SSVEP 2 = ' + str(phase_shift_1_2) + ' degrees')



