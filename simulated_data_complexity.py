#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 16:22:42 2023

@author: James Dowsett
"""

from flicker_analysis_package import functions

import numpy as np
import matplotlib.pyplot as plt
import random 
import scipy.stats

##### make simulated data of random noise and a sine wave SSVEP with triggers in noise

length_of_data_in_seconds = 600
sample_rate = 5000

length_of_data = length_of_data_in_seconds * sample_rate


amplitude_of_noise = 20

print('Amplitude of noise = ' + str(amplitude_of_noise))

plt.suptitle('Amplitude of noise = ' + str(amplitude_of_noise))

noise = np.random.rand(length_of_data,) * amplitude_of_noise


amplitude_of_SSVEP = 1

flicker_frequency = 40

period = int(sample_rate/flicker_frequency)

## make one cycle of a sine wave for the simulated SSVEP
time_vector = np.arange(0,period)
one_cycle_sine_wave = np.sin(2 * np.pi * flicker_frequency/sample_rate * time_vector) * amplitude_of_SSVEP


# function for comparing the fit of any SSVEP using a subset of triggers to the true SSVEP (using all triggers)
def fit_abs(SSVEP_1, SSVEP_2):
    absolute_difference = np.abs(SSVEP_1 - SSVEP_2)
    mean_abs_difference = absolute_difference.mean()
    return mean_abs_difference


## function to compare the SSVEP with a slightly bigger SSVEP using the above fit function

def fit_compare_bigger(running_SSVEP, true_SSVEP):
    
    bigger_running_SSVEP = running_SSVEP * 1.0001
    
    if fit_abs(bigger_running_SSVEP,true_SSVEP) < fit_abs(running_SSVEP,true_SSVEP):
        return True
    else:
        return False
    


for condition_count in range(0,2):
    
    if condition_count == 0:
        condition = 'variation'
     #   plt.subplot(1,2,1)
     #   plt.title('variation')
    elif condition_count == 1:
        condition =  'null'
     #   plt.subplot(1,2,2)
     #   plt.title('null')

    print('\n' + condition + '\n')

    ##  make entire time series of repeated SSVEPs with varying amounts of amplitude variation, or the null condition where every SSVEP is the same
    true_signal = np.zeros([length_of_data,])
    
    k = 0
    while k < length_of_data:
        if condition == 'null': # in the null condition, there is no variation in the size of the SSVEP
            scaling_factor = 1
        elif condition == 'variation': # in the variation condition, the SSVEP is a random size each time
            scaling_factor = random.random()*2

        true_signal[k:k+period] = one_cycle_sine_wave * scaling_factor
        k = k + period
    
    ## make triggers
    triggers = np.arange(0,length_of_data-period,period) ## 
    
    # make time series of triggers to plot and check they are correct
    # trigger_time_series = np.zeros([length_of_data,])
    # for trigger in triggers:
    #     trigger_time_series[trigger] = 1
    
    
    ## add the true signal and the noise
    data = true_signal + noise
    
    # plt.plot(data)
    #plt.plot(true_signal)
    # plt.plot(trigger_time_series)
    
    
    # make the true SSVEP
    true_SSVEP = functions.make_SSVEPs(data, triggers, period)
    
   # plt.plot(true_SSVEP,'b', label = 'True SSVEP')
    
     
    
    ## randomly select a subset of triggers and make an SSVEP, and a random control SSVEP to calculate signal to noise ratio
    
    numbers_of_triggers_to_use = np.arange(1,100,2)
    
    mean_scores_subset = np.zeros([len(numbers_of_triggers_to_use),])
    mean_scores_shuffled = np.zeros([len(numbers_of_triggers_to_use),])
    
    std_scores_subset = np.zeros([len(numbers_of_triggers_to_use),])
    std_scores_shuffled = np.zeros([len(numbers_of_triggers_to_use),])
    
    mean_difference_scores = np.zeros([len(numbers_of_triggers_to_use),])
    std_difference_scores = np.zeros([len(numbers_of_triggers_to_use),])
    
    num_of_triggers_count = 0
    for num_triggers in numbers_of_triggers_to_use:
        
        print('Making SSVEPs with ' + str(num_triggers) + ' triggers')
    
        num_loops = 10000
    
        scores = np.zeros([num_loops,])
        random_scores = np.zeros([num_loops,])
    
        for loop in range(0,num_loops):
            
            random.shuffle(triggers)
    
            random_triggers = triggers[0:num_triggers]
                
            subset_SSVEP = functions.make_SSVEPs(data, random_triggers, period)
            
           # plt.plot(subset_SSVEP,'m', label = 'Subset SSVEP')
            
            shuffled_subset_SSVEP = functions.make_SSVEPs_random(data, random_triggers, period)
            
           # plt.plot(shuffled_subset_SSVEP,'k', label = 'Shuffled Subset SSVEP')
    
            scores[loop] = np.ptp(subset_SSVEP)  #fit_abs(subset_SSVEP, true_SSVEP)
            
            random_scores[loop] = np.ptp(shuffled_subset_SSVEP)  #fit_abs(shuffled_subset_SSVEP, true_SSVEP)
        

        mean_scores_subset[num_of_triggers_count] = scores.mean()
        mean_scores_shuffled[num_of_triggers_count] = random_scores.mean()

        std_scores_subset[num_of_triggers_count] = np.std(scores)
        std_scores_shuffled[num_of_triggers_count] = np.std(random_scores)

        mean_difference_scores[num_of_triggers_count] = (random_scores - scores).mean()
        std_difference_scores[num_of_triggers_count] = np.std((random_scores - scores))
    
    
        num_of_triggers_count += 1
    
    
    ## plots
    
    plt.subplot(1,2,1)
    plt.title('Mean amplitude')
    plt.plot(numbers_of_triggers_to_use,mean_scores_subset, label = 'SSVEPs from a subset of triggers: ' + condition)
    plt.plot(numbers_of_triggers_to_use,mean_scores_shuffled, label = 'Shuffled data: ' + condition)
 
    plt.subplot(1,2,2)
    plt.title('Std amplitude')
    plt.plot(numbers_of_triggers_to_use,std_scores_subset, label = 'SSVEPs from a subset of triggers: ' + condition)
    plt.plot(numbers_of_triggers_to_use,std_scores_shuffled, label = 'Shuffled data: ' + condition)
 
    ### make smallest and largest SSVEP
    
   #  ## randomly split the triggers into two and make an SSVEP from each
   #  random.shuffle(triggers)
    
   #  random_half_triggers_1 = triggers[0:int(len(triggers)/2)]
   #  random_half_triggers_2 = triggers[int(len(triggers)/2):len(triggers)]
    
   #  random_SSVEP_1 = functions.make_SSVEPs(data, random_half_triggers_1, period)
   #  random_SSVEP_2 = functions.make_SSVEPs(data, random_half_triggers_2, period)

   #  # compare the two SSVEPs, find the biggest
   #  if fit_compare_bigger(random_SSVEP_1, random_SSVEP_2):
   #      running_SSVEP = np.copy(random_SSVEP_1)
   #      bigger_triggers = np.copy(random_half_triggers_1)
   #      smaller_triggers = np.copy(random_half_triggers_2)
   #  else:
   #      running_SSVEP = np.copy(random_SSVEP_2)
   #      bigger_triggers = np.copy(random_half_triggers_2)
   #      smaller_triggers = np.copy(random_half_triggers_1)
   
   #  # make an SSVEP with the bigger triggers  
   #  running_SSVEP = functions.make_SSVEPs(data, bigger_triggers, period)
    
   #  # now loop many times, randomly swapping one trigger from the bigger set, 
   #  # with a random trigger from the smaller set
    
   #  num_loops = 10000
    
   #  for loop in range(0,num_loops):
        
   #      print('Loop ' + str(loop) + ' of ' + str(num_loops))
        
   #      temp_bigger_trigs = np.copy(bigger_triggers)
        
   #      # select a random trigger from the other triggers to swap with
       
   #      bigger_trigger_to_swap = random.randint(0, len(bigger_triggers)-1)
   #      smaller_trigger_to_swap = random.randint(0, len(smaller_triggers)-1)
        
   #      temp_bigger_trigs[bigger_trigger_to_swap] = smaller_triggers[smaller_trigger_to_swap]

   #      temp_SSVEP = functions.make_SSVEPs(data, temp_bigger_trigs, period)

   #      if fit_compare_bigger(running_SSVEP, temp_SSVEP):
            
   #          smaller_triggers[smaller_trigger_to_swap] = bigger_triggers[bigger_trigger_to_swap]
   #          bigger_triggers[bigger_trigger_to_swap] = temp_bigger_trigs[bigger_trigger_to_swap]
            
   #          running_SSVEP = functions.make_SSVEPs(data, bigger_triggers, period)
            
   #          print('Bigger')
            
            
   #  bigger_SSVEP =  functions.make_SSVEPs(data, bigger_triggers, period)     

   #  plt.plot(bigger_SSVEP,'r', label = 'Bigger SSVEP')

   # # correlation_split = functions.compare_SSVEPs_split(data, triggers, period)
    
   
   
  

plt.legend()
