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


num_loops = 1000

## Function for loading data from one channel using electrode number
def load_data(file_name, electrode):
    
    import numpy as np
    import mne
    import os
    
    # read the EEG data with the MNE function
    raw = mne.io.read_raw_brainvision(file_name + '.vhdr')
    
    #channel_names = raw.info.ch_names
    
    # Load data
    print('  ')
    print('Loading Electrode data ...')
    print('  ')
    
    
    data = np.array(raw[electrode,:], dtype=object) 
    data = data[0,]
    data = data.flatten()


    print('Saving ...')

    electrode_data_file_name = file_name + '_' + raw.ch_names[electrode] +'_data'
    
    # save as a numpy array
    np.save(os.path.join(file_name, electrode_data_file_name), data)
    
    
    print('Done')
    
## Function for loading data from one channel using electrode name
def load_data_electrode_name(file_name, electrode_name):
    
    import numpy as np
    import mne
    import os
    
    # read the EEG data with the MNE function
    raw = mne.io.read_raw_brainvision(file_name + '.vhdr')
    
    channel_names = raw.info.ch_names
    
    electrode = channel_names.index(electrode_name)
    
    # Load data
    print('  ')
    print('Loading Electrode data ...')
    print('  ')
    
    data = np.array(raw[electrode,:], dtype=object) 
    data = data[0,]
    data = data.flatten()
    
    print('Saving ...')

    electrode_data_file_name = file_name + '_' + raw.ch_names[electrode] +'_data'
    
    # save as a numpy array
    np.save(os.path.join(file_name, electrode_data_file_name), data)
    
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
def make_SSVEPs(data, all_triggers, period):
    import numpy as np

    segment_matrix = np.zeros([len(all_triggers), period]) # empty matrix to put segments into
    
    seg_count = 0 # keep track of the number of segments
    
    # loop through all triggers and put the corresponding segment of data into the matrix
    for trigger in all_triggers:
        
        # select a segment of data the lenght of the flicker period, starting from the trigger time 
        segment =  data[trigger:trigger+period] 
        
        segment_matrix[seg_count,:] = segment
    
        seg_count += 1
    
    SSVEP = segment_matrix.mean(axis=0) # average to make SSVEP
    
    SSVEP = SSVEP - SSVEP.mean() # baseline correct
    
    return SSVEP

##Function to make SSVEPs with offset (use triggers & average)
def make_SSVEPs_offset(data, triggers, period, offset):
    import numpy as np

    segment_matrix = np.zeros([len(triggers), period]) # empty matrix to put segments into
    
    seg_count = 0 # keep track of the number of segments
    
    # loop through all triggers and put the corresponding segment of data into the matrix
    for trigger in triggers:
        
        # select a segment of data the lenght of the flicker period, starting from the trigger time 
        segment =  data[trigger-offset:trigger+period-offset] 
        
        segment_matrix[seg_count,:] = segment
    
        seg_count += 1
    
    SSVEP = segment_matrix.mean(axis=0) # average to make SSVEP
    
    SSVEP = SSVEP - SSVEP.mean() # baseline correct
    
    return SSVEP

##Function to make SSVEPs (randomly shuffle data in each segment, then average - way of calculating signal-to-noise ratio)
def make_SSVEPs_random(data, all_triggers, period, num_loops):
    ## make SSVEP with all segments
    
    import numpy as np
    import matplotlib.pyplot as plt
    import random 

    segment_matrix = np.zeros([len(all_triggers), period]) # empty matrix to put segments into
    
    seg_count = 0 # keep track of the number of segments
    
    # loop through all triggers and put the corresponding segment of data into the matrix
    for trigger in all_triggers:
        
        # select a segment of data the lenght of the flicker period, starting from the trigger time 
        segment =  data[trigger:trigger+period] 
        
        segment_matrix[seg_count,:] = segment
    
        seg_count += 1
    
    SSVEP = segment_matrix.mean(axis=0) # average to make SSVEP
    
    SSVEP = SSVEP - SSVEP.mean() # baseline correct
    
    
    # permutation tests on noise
    
    
    random_amplitudes = np.zeros([num_loops,])
    
    for loop in range(0,num_loops):
        
       # print(loop)
        # make random SSVEP 
        
        shuffled_segment_matrix =  np.zeros([len(all_triggers), period])  
        
        # loop through all triggers and put the corresponding segment of data into the matrix
        seg_count = 0 # keep track of the number of segments
        
        for trigger in all_triggers:
            
            segment =  data[trigger:trigger+period] 
        
            random.shuffle(segment) # randomly shuffle the data points
            
            shuffled_segment_matrix[seg_count,:] = segment
            
            seg_count += 1
        
        random_SSVEP = shuffled_segment_matrix.mean(axis=0) # average to make SSVEP
        
        random_SSVEP = random_SSVEP - random_SSVEP.mean() # baseline correct
        
        random_amplitudes[loop] = np.ptp(random_SSVEP)
    
    
    plt.plot(random_SSVEP,'k') # plot the last random shuffle, just to see
    
    plt.plot(SSVEP,'b') # plot the true SSVEP
    
    true_amplitude = np.ptp(SSVEP)
    
    print('True amplitude = ', true_amplitude)
    
    average_noise = random_amplitudes.mean()
    
    print('Amplitude noise = ', average_noise)
    
    std_noise = np.std(random_amplitudes)
    
    print('Standard Deviation noise = ', std_noise)
    
    Z_score  = (true_amplitude-average_noise) / std_noise
    
    print('Z_score = ', Z_score)
    
##Function for only random SSVEPs and z score
def randomSSVEPs_zscore(SSVEP, data, all_triggers, period, num_loops, offset):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import random
    
    random_amplitudes = np.zeros([num_loops,])
    
    for loop in range(0,num_loops):
        
        print(loop)
        # make random SSVEP 
        
        shuffled_segment_matrix =  np.zeros([len(all_triggers), period])  
        
        # loop through all triggers and put the corresponding segment of data into the matrix
        seg_count = 0 # keep track of the number of segments
        
        for trigger in all_triggers:
            
            segment =  data[trigger-offset:trigger+period-offset] 
        
            random.shuffle(segment) # randomly shuffle the data points
            
            shuffled_segment_matrix[seg_count,:] = segment
            
            seg_count += 1
        
        random_SSVEP = shuffled_segment_matrix.mean(axis=0) # average to make SSVEP
        
        random_SSVEP = random_SSVEP - random_SSVEP.mean() # baseline correct
        
        random_amplitudes[loop] = np.ptp(random_SSVEP)
    
    
    plt.plot(random_SSVEP,'k') # plot the last random shuffle, just to see
    
    plt.plot(SSVEP,'b') # plot the true SSVEP
    
    true_amplitude = np.ptp(SSVEP)
    
    print('True amplitude = ', true_amplitude)
    
    average_noise = random_amplitudes.mean()
    
    print('Amplitude noise = ', average_noise)
    
    std_noise = np.std(random_amplitudes)
    
    print('Standard Deviation noise = ', std_noise)
    
    Z_score  = (true_amplitude-average_noise) / std_noise
    
    print('Z_score = ', Z_score)



##Function for linear interpolation of trigger artefacts (plot SSVEP showing before and after in this function)

##Function for making induced FFT
#Desc: Segment data into segments of a given length, do an FFT on each segment and then average the FFTs.

##Function for making evoked FFT
#Desc: Segment data into non-overlapping segments of a given length, each time locked to a trigger. Then average and do an FFT on the average.


###ANALYSIS FUNCTIONS that require making SSSVEPs

##Permutation tests on two conditions
#Desc: Make two SSVEPs by randomly assigning segments from condition 1 and condition 2 to two make two groups, and compare the difference in amplitude. Repeat this may times (e.g. 1000) and create a distribution of amplitude differences, which can be compared to the true difference.

##Permutation tests on signal-to-noise
#Desc: Make SSVEPs using a random subset of triggers (start with a small number), repeat many times (e.g.1000). Then repeat entire test with a different number of segments. Progressively increase number of segments and keep track of the resulting SSVEP.

##Simulated SSVEPs (for data without flicker)
#Desc: Add a simulated SSVEP (e.g. a sine wave) to the data and create a numpy array of “trigger times”, such that the data can be used with the above functions.

##Plots (for plotting example SSVEPs)


##### ANALYSIS FUNCTIONS ON SSVEPs that are already averaged

def compareSSVEPs(SSVEP_1, SSVEP_2):
    
    import numpy as np
    import matplotlib.pyplot as plt

    amplitude_difference = np.ptp(SSVEP_1) - np.ptp(SSVEP_2)

    # check if SSVEPs are the same length
    
    if len(SSVEP_1) == len(SSVEP_2):
        
        correlation = np.corrcoef(SSVEP_1, SSVEP_2)    # Pearson correlation

    else: # if not the same length, time-warp the shorter of the two so they are the same length, call the new time warped wave: new_SSVEP 
    
        if len(SSVEP_1) < len(SSVEP_2):
            
            longer_SSVEP = np.copy(SSVEP_2)
            shorter_SSVEP = np.copy(SSVEP_1)
            
        elif len(SSVEP_1) > len(SSVEP_2):
            
            longer_SSVEP = np.copy(SSVEP_1)
            shorter_SSVEP = np.copy(SSVEP_2)

        ### time warp: 
            
        ## make a much longer version of the shorter SSVEP by evenly spacing each data point and linear interpolation of the data points in between

        length_temp_SSVEP = 1000 # the length of the temporary waveform, which will be re-scaled
        
        temp_SSVEP = np.zeros([length_temp_SSVEP,]) # empty matrix for the long re-scaled SSVEP
        
        
        steps_to_use = np.linspace(0,length_temp_SSVEP,len(shorter_SSVEP)) # evenly spaced time points, one for each time point of the shorter SSVEP
        steps_to_use = np.round(steps_to_use) # round these time points to integer values
        
        for k in range(0,len(longer_SSVEP)-1): # for every time point in the longer SSVEP
            t = int(steps_to_use[k]) # get the data value of one time point
            t_2 = int(steps_to_use[k+1]) # and of the following time point
            temp_SSVEP[t:t_2] = np.linspace(shorter_SSVEP[k],shorter_SSVEP[k+1],len(temp_SSVEP[t:t_2])) # add these data points, and linear interpolation between them, to the temp SSVEP
           
        
        
        new_SSVEP = np.zeros([len(longer_SSVEP),]) # empty array to put the new values into, the length of the longer SSVEP
        
        
        steps_to_use = np.linspace(0,length_temp_SSVEP,len(longer_SSVEP)) # evenly spaced time points,  one for each time point of the longer SSVEP
        steps_to_use = np.round(steps_to_use) # round these time points to integer values
        
        for k in range(0,len(shorter_SSVEP)-1): # get the values from the longer SSVEP, and use them to make the new SSVEP
            t = int(steps_to_use[k])   
            new_SSVEP[k] = temp_SSVEP[t]
        
        
        new_SSVEP[-1] = temp_SSVEP[length_temp_SSVEP-1] # put the last time point in seperatly


        correlation = np.corrcoef(longer_SSVEP, new_SSVEP)    # Pearson correlation of the longer SSVEP and the time warped shorter SSVEP

    
    return correlation



