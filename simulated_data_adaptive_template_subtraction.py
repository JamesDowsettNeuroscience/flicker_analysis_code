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

from flicker_analysis_package import functions

trial_time = 300 # total time in seconds

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



triggers = np.arange(0, len(simulated_SSVEP_data)-period, period) # make triggers, stop one period length before the end


## load one segment of typical walking artefact
artefact_segment = np.load('example_artefact_segment.npy')

artefact_segment = artefact_segment - artefact_segment.mean() # baseline correct

number_of_walking_cycles =  int(len(simulated_SSVEP_data)/len(artefact_segment))  # int number of times the walking artefact will fit into the data
walking_artefact_data = np.zeros([trial_time * sample_rate])
walking_artefact_data[0:number_of_walking_cycles*len(artefact_segment)] = np.tile(artefact_segment,number_of_walking_cycles)


## combine all data

data = simulated_SSVEP_data + walking_artefact_data

# plot raw data

trigger_time_series = np.zeros([len(simulated_SSVEP_data)],)
for trigger in triggers:
    trigger_time_series[trigger] = 1


plt.figure()

# plt.plot(trigger_time_series)
    
# plt.plot(simulated_SSVEP_data)

# plt.plot(walking_artefact_data)

# plt.plot(data)


## make SSVEP

SSVEP = functions.make_SSVEPs(data,triggers,period)


plt.plot(one_cycle_SSVEP)

plt.plot(SSVEP)




########### random walk template construction #################
        
# random_walk_template = np.zeros([len(data)],)

# k = 0

# value = 0

# while k < len(data):
    
#     random_walk_template[k] = data[k] + ((random.random() * 10) - 5)
    
#     k+=1



####################  adaptive template subtraction with random segments from the rest of the data  ######################

num_cycles_for_template = 10
num_templates = 20

clean_SSVEP = functions.make_SSVEP_artefact_removal(data, triggers, period, num_cycles_for_template, num_templates)

plt.plot(clean_SSVEP,'g')




##################### test with just subtracting random segments

# numbers_of_segs_to_test = np.arange(10,100,10)

# differences_in_amplitude = np.zeros([len(numbers_of_segs_to_test),])

# count = 0


# for number_of_segs in numbers_of_segs_to_test:

# number_of_segs = 2   

# segment_matrix = np.zeros([len(simulated_triggers),period])
# seg_count = 0


# for trigger in simulated_triggers:
    
#     segment = simulated_SSVEP_data[trigger:trigger+period]
    
#     random_segment_matrix = np.zeros([number_of_segs,period])
#     for random_seg_count in range(0,number_of_segs):
#         random_position = random.randint(0, len(simulated_SSVEP_data)-period) # pick a randon point in the data
#         random_segment = simulated_SSVEP_data[random_position:random_position+period]
#         random_segment_matrix[random_seg_count,:] = random_segment
    
#     averaged_random_segs = random_segment_matrix.mean(axis=0)
    
#     segment = segment - averaged_random_segs
    
#     segment_matrix[seg_count,:] = segment
#     seg_count += 1
    
# SSVEP = segment_matrix.mean(axis=0)

# # differences_in_amplitude[count] = np.ptp(SSVEP) - np.ptp(one_cycle_SSVEP)

# # count += 1

# plt.plot(SSVEP)

# plt.plot(one_cycle_SSVEP)
    
        
# plt.plot(numbers_of_segs_to_test,differences_in_amplitude)   
    


