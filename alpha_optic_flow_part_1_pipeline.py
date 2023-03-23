#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:48:54 2023

@author: James Dowsett
"""

### alpha optic flow part 1 analysis pipeline ######


from flicker_analysis_package import functions

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

import random

### information about the experiment:

path = '/media/james/Expansion/alpha_optic_flow/experiment_1_data/'

condition_names = ('Eyes Track, head fixed', 'Head track, Eyes fixed', 'Both head and eyes track', 'Eyes Track, head fixed - control', 'Head track, Eyes fixed - control', 'Both head and eyes track - control')

electrode_names = ('O1', 'O2', 'P3', 'P4', 'P7', 'P8', 'REF2', 'EOG')

sample_rate = 1000

num_subjects = 20

length = 1 # length of FFT in seconds


subject = 1

for block in(1,2):
    
    print('\n Block ' + str(block) + '\n')
    
    condition = 1
    
    
    trial_name = 'subject_' + str(subject) + '_block_' + str(block) + '_condition_' + str(condition) 
    
    all_data = np.load(path + trial_name + '_all_data.npy')
    
    triggers = np.load(path + trial_name + '_all_triggers.npy')
    
    
    ## check for eye blinks
    EOG_data = all_data[7,:]
    
    # trigger_time_series = np.zeros([len(EOG_data),])
    # for trigger in triggers:
    #     trigger_time_series[trigger] = 0.0001
    
    # plt.plot(trigger_time_series)
    # plt.plot(EOG_data)
    
    good_triggers = []
    
    for trigger in triggers:
        if np.ptp(EOG_data[trigger-100:trigger+100]) < 0.00005:
            good_triggers.append(trigger)
    
    # good_trigger_time_series = np.zeros([len(EOG_data),])
    # for trigger in good_triggers:
    #     good_trigger_time_series[trigger] = 0.0001
        
    # plt.plot(good_trigger_time_series)
    
       
    ## sort triggers by period into different frequency bins
    
    freq_bins = np.arange(72,125,5) # make frequency bins
    
    diff_triggers = np.diff(good_triggers)     
    
    for current_bin in range(0,len(freq_bins)):
        
        plt.figure(current_bin)
       
        triggers_for_this_bin = []
        
        for trig_count in range(0,len(diff_triggers)):
            
            if diff_triggers[trig_count] < 128:
                
                # get the frequency bin with the period closest to the period of the flicker trigger
                freq_bin = np.argmin(np.abs(freq_bins - diff_triggers[trig_count]))
    
                if freq_bin == current_bin:
                    triggers_for_this_bin.append(good_triggers[trig_count])
                    
        num_triggers_for_bin = len(triggers_for_this_bin)
        print(str(num_triggers_for_bin) + ' segments ' + str(freq_bins[current_bin]) + ' ms = ' + str(np.round(sample_rate/freq_bins[current_bin],2)) + ' Hz')
        
        plt.suptitle(str(num_triggers_for_bin) + ' segments ' + str(freq_bins[current_bin]) + ' ms = ' + str(np.round(sample_rate/freq_bins[current_bin],2)) + ' Hz')
        
        
        
        
        REF_2_data = all_data[6,:]
        
        for electrode in range(0,6):
            
            plt.subplot(2,3,electrode+1)
            plt.title(electrode_names[electrode])
            
            data = all_data[electrode,:]
            
            data = data - (REF_2_data/2) # re-reference
            
            SSVEP = functions.make_SSVEPs(data, triggers_for_this_bin, freq_bins[current_bin])
            
            #plt.plot(data, label = electrode_names[electrode])
            plt.plot(SSVEP, label = block)
        
        plt.legend()