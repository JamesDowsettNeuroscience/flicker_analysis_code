#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:45:32 2022

@author: James Dowsett
"""

import numpy as np
import matplotlib.pyplot as plt

from flicker_analysis_package import functions

period = 25

time_vector = np.arange(0,period)

flicker_frequency = 40
sample_rate = 1000

phase_shift = 0

one_cycle_SSVEP = np.sin(2 * np.pi * flicker_frequency/sample_rate * (time_vector-phase_shift)) 


# repeat the single cycle at the correct frequency

number_of_flickers = int((100*sample_rate)/period) # number of times the simulated flicker will repeat in 100 seconds

# empty array to put the simulated SSVEP into
simulated_SSVEP_data = np.zeros([100 * sample_rate])

# use tile to repeat the basic SSVEP
simulated_SSVEP_data[0:number_of_flickers*period] = np.tile(one_cycle_SSVEP,number_of_flickers )

simulated_triggers = np.arange(0, len(simulated_SSVEP_data)-period, period) # make triggers, stop one period length before the end

num_loops = 1000

num_subjects = 1


noise_values_to_test = np.arange(10,2010,100)


grand_average_absolute_phase_shifts = np.zeros([len(noise_values_to_test),])

for noise_level in range(0,len(noise_values_to_test)):


    noise_amplitude = noise_values_to_test[noise_level]  

    average_absolute_phase_shifts = np.zeros([num_subjects,])

    for subject in range(0,num_subjects):

        noise = np.random.rand(len(simulated_SSVEP_data),) * noise_amplitude
        
        simulated_data = simulated_SSVEP_data + noise
        
        absolute_phase_shifts = np.zeros([num_loops,])
        
        for loop in range(0,num_loops):

            phase_shift = functions.phase_shift_SSVEPs_split(simulated_data, simulated_triggers, period)

            absolute_phase_shift = np.abs(phase_shift)
            
            absolute_phase_shifts[loop] = absolute_phase_shift
            
            
        
        average_absolute_phase_shift = absolute_phase_shifts.mean()
        
        
        average_absolute_phase_shifts[subject] = average_absolute_phase_shift
       
    grand_average = average_absolute_phase_shifts.mean()
    grand_average_absolute_phase_shifts[noise_level] = grand_average
    
    print('Noise = ' + str(noise_amplitude) + '  Average Phase shift = ' + str(np.round(grand_average)) + ' degrees')
    
    
plt.plot(noise_values_to_test,grand_average_absolute_phase_shifts,'g')
    
plt.xlabel('Noise amplitude (in units of signal amplitude)')
plt.ylabel('Average absolute phase shift (degrees)')

plt.title('Simulation of 50% split of 100 seconds of 40 Hz flicker, ' + str(num_subjects) + ' subjects')

plt.plot(np.arange(0, 2000),np.ones([2000,])*90,  '--k')   

    # plt.plot(SSVEP_1)
    # plt.plot(SSVEP_2)
    
    # plt.title('Phase shift = ' + str(np.round(phase_shift)) + ' degrees')
    
    
    