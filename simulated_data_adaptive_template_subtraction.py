#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 14:19:31 2022

@author: James Dowsett
"""


#### Code to check the loss of quality in the final SSVEP, as a result of subtracting

import numpy as np
import matplotlib.pyplot as plt

import random

trial_time = 2 # total time in seconds

period = 111

time_vector = np.arange(0,period)

flicker_frequency = 9
sample_rate = 1000

phase_shift = 0

SSVEP_amplitude = 1


# one_cycle_SSVEP = SSVEP_amplitude * np.sin(2 * np.pi * flicker_frequency/sample_rate * (time_vector-phase_shift)) 

one_cycle_SSVEP = np.load('example_9Hz_SSVEP.npy')  
    
# repeat the single cycle at the correct frequency

number_of_flickers = int((trial_time*sample_rate)/period) # number of times the simulated flicker will repeat in 100 seconds

# empty array to put the simulated SSVEP into
simulated_SSVEP_data = np.zeros([trial_time * sample_rate])

# use tile to repeat the basic SSVEP
simulated_SSVEP_data[0:number_of_flickers*period] = np.tile(one_cycle_SSVEP,number_of_flickers )



simulated_triggers = np.arange(0, len(simulated_SSVEP_data)-period, period) # make triggers, stop one period length before the end


# plot raw data

trigger_time_series = np.zeros([len(simulated_SSVEP_data)],)
for trigger in simulated_triggers:
    trigger_time_series[trigger] = 1


# plt.plot(trigger_time_series)
    
# plt.plot(simulated_SSVEP_data)


##

# numbers_of_segs_to_test = np.arange(10,100,10)

# differences_in_amplitude = np.zeros([len(numbers_of_segs_to_test),])

# count = 0


# for number_of_segs in numbers_of_segs_to_test:

number_of_segs = 2   

segment_matrix = np.zeros([len(simulated_triggers),period])
seg_count = 0


for trigger in simulated_triggers:
    
    segment = simulated_SSVEP_data[trigger:trigger+period]
    
    random_segment_matrix = np.zeros([number_of_segs,period])
    for random_seg_count in range(0,number_of_segs):
        random_position = random.randint(0, len(simulated_SSVEP_data)-period) # pick a randon point in the data
        random_segment = simulated_SSVEP_data[random_position:random_position+period]
        random_segment_matrix[random_seg_count,:] = random_segment
    
    averaged_random_segs = random_segment_matrix.mean(axis=0)
    
    segment = segment - averaged_random_segs
    
    segment_matrix[seg_count,:] = segment
    seg_count += 1
    
SSVEP = segment_matrix.mean(axis=0)

# differences_in_amplitude[count] = np.ptp(SSVEP) - np.ptp(one_cycle_SSVEP)

# count += 1

plt.plot(SSVEP)

plt.plot(one_cycle_SSVEP)
    
        
# plt.plot(numbers_of_segs_to_test,differences_in_amplitude)   
    


