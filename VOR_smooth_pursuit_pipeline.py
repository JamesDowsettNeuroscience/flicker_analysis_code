#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 12:38:48 2023

@author: James Dowsett
"""

############ analysis pipeline for VOR-smooth pursuit ficker experiment ##################

import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

### information about the experiment:

path = '/home/james/Active_projects/VOR_EEG/' # put path here

with open(path + "chan_names", "rb") as fp:   # Unpickling electrode names
    electrode_names = pickle.load(fp)

condition_names = ('Eyes Track, head fixed', 'Head track, Eyes fixed', 'Both head and eyes track', 'Eyes Track, head fixed - control', 'Head track, Eyes fixed - control', 'Both head and eyes track - control')

frequency_names = ('10 Hz', '40 Hz')

sample_rate = 5000

num_subjects = 2

period_10Hz = 500 # period of flicker in data points
period_40Hz = 125 # period of flicker in data points

length = 1 # length of FFT in seconds

## trigger times for linear interpolation
trig_1_time = -3
trig_2_time = 48
trig_length = 6



#######################################


subject = 1

frequency = 0 # 0 = 10Hz, 1 = 40Hz

for condition_count in range(0,6):
    
    if frequency == 0:
        condition = condition_count+1
    elif frequency == 1:
        condition = condition_count+7 
        
    print(condition_names[condition_count])
    

    ## load all data
    all_data = np.load(path + 'S' + str(subject) + '_' + str(condition) + '_all_data.npy')
    
    ## get EOG data
    HEOG_data = all_data[31,:]
    VEOG_data = all_data[30,:]

    HEOG_data = HEOG_data - HEOG_data.mean() # baseline correct
    VEOG_data = VEOG_data - VEOG_data.mean()
    
    
    ## filter EOG data
    
    nyquist = sample_rate/2

    # low pass HEOG
    cutoff = 1 # cutoff in Hz
    Wn = cutoff/nyquist
    b, a = scipy.signal.butter(3, Wn, 'low')
    HEOG_data = scipy.signal.filtfilt(b, a, HEOG_data)
    
    # high pass HEOG
    cutoff = 0.05 # cutoff in Hz
    Wn = cutoff/nyquist
    b, a = scipy.signal.butter(3, Wn, 'high')
    HEOG_data = scipy.signal.filtfilt(b, a, HEOG_data)

    # low pass VEOG
    cutoff = 5 # cutoff in Hz
    Wn = cutoff/nyquist
    b, a = scipy.signal.butter(3, Wn, 'low')
    VEOG_data = scipy.signal.filtfilt(b, a, VEOG_data)
    
    # high pass VEOG
    cutoff = 1 # cutoff in Hz
    Wn = cutoff/nyquist
    b, a = scipy.signal.butter(3, Wn, 'high')
    VEOG_data = scipy.signal.filtfilt(b, a, VEOG_data)

    

    ## load triggers
    all_flicker_triggers = np.load(path + 'S' + str(subject) + '_' + str(condition) + '_flicker_triggers.npy')
    
    all_movement_triggers = np.load(path + 'S' + str(subject) + '_' + str(condition) + '_movement_triggers.npy')
    
    # make trigger time series
    flicker_trigger_time_series = np.zeros([all_flicker_triggers[-1]+1,])
    movement_trigger_time_series = np.zeros([all_movement_triggers[-1]+1,])
    
    for trigger in all_flicker_triggers:
        flicker_trigger_time_series[trigger] = 0.0001
        
    for trigger in all_movement_triggers:
        movement_trigger_time_series[trigger] = 0.0002
       
    plt.figure()
    plt.title(condition_names[condition-1])
    plt.plot(flicker_trigger_time_series)
    plt.plot(movement_trigger_time_series)
    
    plt.plot(HEOG_data)
    plt.plot(VEOG_data)
    
    
    ## sort triggers 
    good_flicker_triggers_list = []
    
    
    wait_time = 5000 # one second wait time for the moving condition

    for trigger in all_flicker_triggers:
        
        trigger_distance_from_movement = all_movement_triggers - trigger
        
        # select the next movement trigger in the future
        nearest_movement_trigger = min(trigger_distance_from_movement[trigger_distance_from_movement>0])
        
        # if next movement trigger is more than one second away
        if nearest_movement_trigger > wait_time: 
            
            # check for eye blinks
            VEOG_segment = VEOG_data[trigger-500:trigger+500]
            
            if np.ptp(VEOG_segment) < 0.00005:
            
                good_flicker_triggers_list.append(trigger)
        
        
        
        
    # plot time series of good triggers   
    good_flicker_triggers_time_series = np.zeros([all_flicker_triggers[-1]+1,])
    for trigger in good_flicker_triggers_list:
        good_flicker_triggers_time_series[trigger] = 0.00015
     
        
    plt.plot(good_flicker_triggers_time_series)
    
    
    
    
    
    