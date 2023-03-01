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

length_of_data_in_seconds = 10 #600
sample_rate = 5000

length_of_data = length_of_data_in_seconds * sample_rate


num_triggers_to_use = 1000

amplitude_of_noise = 0

print('Amplitude of noise = ' + str(amplitude_of_noise))

noise = np.random.rand(length_of_data,) * amplitude_of_noise


amplitude_of_SSVEP = 1

flicker_frequency = 40

period = int(sample_rate/flicker_frequency)

## make one cycle of a sine wave for the simulated SSVEP
time_vector = np.arange(0,period)
one_cycle_sine_wave = np.sin(2 * np.pi * flicker_frequency/sample_rate * time_vector) * amplitude_of_SSVEP



for condition_count in range(0,2):
    
    if condition_count == 0:
        condition = 'variation'
        plt.subplot(1,2,1)
        plt.title('variation')
    elif condition_count == 1:
        condition =  'null'
        plt.subplot(1,2,2)
        plt.title('null')

    ##  make entire time series of repeated SSVEPs with varying amounts of amplitude variation, or the null condition where every SSVEP is the same
    true_signal = np.zeros([length_of_data,])
    
    k = 0
    while k < length_of_data:
        if condition == 'null':
            scaling_factor = 1
        elif condition == 'variation':
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
    
    

    SSVEP = functions.make_SSVEPs(data, triggers, period)
    
    
    if condition == 'null':
        plt.plot(SSVEP,'b', label = condition)
    elif condition == 'variation':
        plt.plot(SSVEP,'r', label = condition)
        
    
    ### make smallest and largest SSVEP
    
    random.shuffle(triggers)

    start_num = 10


    small_triggers_count = start_num
    big_triggers_count = start_num

    small_triggers = triggers[0:start_num]
    big_triggers = triggers[start_num:start_num*2]
    
    start_num = start_num * 2
    
   
    for trigger_num in range(start_num,len(triggers)):
        
        print('Trigger ' + str(trigger_num) + ' of ' + str(len(triggers)))
        
        trigger = triggers[trigger_num]
        
        running_small_SSVEP = functions.make_SSVEPs(data, small_triggers, period)

        running_big_SSVEP = functions.make_SSVEPs(data, big_triggers, period)

        temp_small_triggers = np.append(small_triggers, trigger_num)

        temp_small_SSVEP = functions.make_SSVEPs(data, temp_small_triggers, period)
        
        if np.ptp(temp_small_SSVEP) < np.ptp(running_small_SSVEP):
            small_triggers = np.append(small_triggers,trigger)
            
        temp_big_triggers = np.append(big_triggers, trigger_num)

        temp_big_SSVEP = functions.make_SSVEPs(data, temp_big_triggers, period)

        if np.ptp(temp_big_SSVEP) > np.ptp(running_big_SSVEP):
            big_triggers = np.append(big_triggers,trigger)


    small_SSVEP = functions.make_SSVEPs(data, small_triggers, period)
    big_SSVEP = functions.make_SSVEPs(data, big_triggers, period)

    plt.plot(small_SSVEP, label = condition + ' small')
    plt.plot(big_SSVEP, label = condition + ' big')

   # correlation_split = functions.compare_SSVEPs_split(data, triggers, period)
    
   
   
  

    plt.legend()
