#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 17:29:18 2023

@author: James Dowsett
"""

# simulate 10 Hz SSVEP in noise.
# SSVEPs are identical in each condition
# attempt to decode, should be 50% decoding accuracy, i.e. chance level


import numpy as np
import matplotlib.pyplot as plt
from flicker_analysis_package import functions


plt.figure()

period = 100

num_flickers = 1200

t = np.arange(0,period)

noise_amplitude = 10

phase_shift = 0

num_replications = 1

num_subjects = 20

repeats_of_experiment = 100


sine_wave_1 = np.sin(2 * np.pi * 1/1000 * 10 * t)
sine_wave_2 = np.sin(2 * np.pi * 1/1000 * 10 * (t+phase_shift))

signal_1 = np.tile(sine_wave_1, num_flickers)
signal_2 = np.tile(sine_wave_2, num_flickers)

triggers_1 = np.arange(0,len(signal_1),period)
triggers_2 = np.arange(0,len(signal_2),period)

numbers_of_loops_to_test = (1,10,100)

for loop_counter in range(0,len(numbers_of_loops_to_test)): # loop for different numbers of repetitions for each decoding attempt
    
    num_loops = numbers_of_loops_to_test[loop_counter]

    average_for_each_experiment = np.zeros([repeats_of_experiment,])
    
    for experiment in range(0, repeats_of_experiment): # repeat the entire simulated experiment multiple times
    
      #  print('\nExperiment ' + str(experiment))
        
        scores = np.zeros([num_subjects,])
        score_ranges = np.zeros([num_subjects,])
        
        for subject in range(0,num_subjects):
            
            data_1 = signal_1 + (np.random.rand(len(signal_1),) * noise_amplitude)
            
            data_2 = signal_2 + (np.random.rand(len(signal_2),) * noise_amplitude)
            
            
            # SSVEP_1 = functions.make_SSVEPs(data_1, triggers_1, period)
            
            # SSVEP_2 = functions.make_SSVEPs(data_2, triggers_2, period)
            
            # plt.plot(SSVEP_1)
            # plt.plot(SSVEP_2)
            
            
            #replication_scores = np.zeros([num_replications,])
            
            #for replication in range(0,num_replications):
     
                
        
            average_percent_correct = functions.decode_correlation(data_1, data_2, triggers_1, triggers_2, period, num_loops)
            
                
                
                #replication_scores[replication] = average_percent_correct
                
          #  print(average_percent_correct)
                
                
                
            
            scores[subject] = average_percent_correct  # replication_scores.mean()
            
            # print('  ')
            # print(np.ptp(replication_scores))
            # print('  ')
            #score_ranges[subject] = np.ptp(replication_scores)
            
            
        #print('Mean score = ' + str(scores.mean()))
        
      #  print('Mean range per subject = ' + str(score_ranges.mean()))
    
        average_for_each_experiment[experiment] = scores.mean()
        
    
    #plt.hist(scores)
    #plt.plot(scores)
    
    plt.subplot(1,3,loop_counter+1)  
    plt.title(str(num_loops) + ' loops')        
  
    
    max_value = max(average_for_each_experiment)
    min_value = min(average_for_each_experiment)
    
    if int(max_value - min_value) > 0:
        num_bins = int(max_value - min_value)
    else:
        num_bins = 10
    
    print('\nNumber of loops = ' + str(num_loops))
    print('min value = ' + str(min_value) + '   max value =  ' + str(max_value) )
    print('Mean score for all experiments = ' + str(average_for_each_experiment.mean()) + '\n')
    
    plt.hist(average_for_each_experiment,num_bins)
  
    
