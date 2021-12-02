#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 12:02:57 2021

@author: prithasen
"""

### PRE-PROCESSING USING MNE

## Function for interpolating bad electrodes

## Function for Laplacian re-referencing

### Loading data, getting and sorting triggers (experiment specicfic)

## Function for loading data from one channel
def load_data(file_name, electrode_name):
    
    import numpy as np
    import mne
    
    # read the EEG data with the MNE function
    raw = mne.io.read_raw_brainvision(file_name + '.vhdr')
    
    channel_names = raw.info.ch_names
    
    electrode = channel_names.index(electrode_name)
    
    # Load data
    print('  ')
    print('Loading EOG data ...')
    print('  ')
    
    
    # extract the EOG data

    VEOG_data = np.array(raw[30,:], dtype=object) 
    VEOG_data = VEOG_data[0,]
    VEOG_data = VEOG_data.flatten()
    
    HEOG_data = np.array(raw[31,:], dtype=object) 
    HEOG_data = HEOG_data[0,]
    HEOG_data = HEOG_data.flatten()
    
    print('Loading electrode data ...')
    print('  ')
    
    
    data = np.array(raw[electrode,:], dtype=object) 
    data = data[0,]
    data = data.flatten()
    
    channel_names = raw.info.ch_names 
    channel_names[60] = 'Fpz' # correct names spelt wrong


    print('Saving ...')

    electrode_data_file_name = file_name + '_' + electrode_name +'_data'
    VEOG_data_file_name = file_name + '_VEOG_data'
    HEOG_data_file_name = file_name + '_HEOG_data'
    
    # save as a numpy array
    np.save(electrode_data_file_name, data)
    np.save(VEOG_data_file_name, VEOG_data)
    np.save(HEOG_data_file_name, HEOG_data)
    
    
    print('Done')
    
##Function for getting triggers from .vmrk file – save as numpy array

##Function for sorting triggers into different conditions/frequencies

###COMMON ANALYSIS FUNCTIONS

##Function for high pass filter
def high_pass_filter(sample_rate, data):
    
    from scipy import signal
    
    high_pass_filter = signal.butter(2, 0.1, 'hp', fs=sample_rate, output='sos')
    high_pass_filtered_data = signal.sosfilt(high_pass_filter, data)
    return high_pass_filtered_data

        
##Function for low pass filter 
def low_pass_filter(sample_rate, data):
    
    from scipy import signal
    
    low_pass_filter = signal.butter(3, 5, 'lp', fs=sample_rate, output='sos')
    low_pass_filtered_data = signal.sosfilt(low_pass_filter, data)
    return low_pass_filtered_data
    

##Function to determine good data

##Function to make SSVEPs (use triggers & average)

##Function to make SSVEPs (randomly shuffle data in each segment, then average - way of calculating signal-to-noise ratio)

##Function for linear interpolation of trigger artefacts (plot SSVEP showing before and after in this function)

##Function for making induced FFT
#Desc: Segment data into segments of a given length, do an FFT on each segment and then average the FFTs.

##Function for making evoked FFT
#Desc: Segment data into non-overlapping segments of a given length, each time locked to a trigger. Then average and do an FFT on the average.

###ANALYSIS FUNCTIONS

##Permutation tests on two conditions
#Desc: Make two SSVEPs by randomly assigning segments from condition 1 and condition 2 to two make two groups, and compare the difference in amplitude. Repeat this may times (e.g. 1000) and create a distribution of amplitude differences, which can be compared to the true difference.

##Permutation tests on signal-to-noise
#Desc: Make SSVEPs using a random subset of triggers (start with a small number), repeat many times (e.g.1000). Then repeat entire test with a different number of segments. Progressively increase number of segments and keep track of the resulting SSVEP.

##Simulated SSVEPs (for data without flicker)
#Desc: Add a simulated SSVEP (e.g. a sine wave) to the data and create a numpy array of “trigger times”, such that the data can be used with the above functions.

##Plots (for plotting example SSVEPs)














