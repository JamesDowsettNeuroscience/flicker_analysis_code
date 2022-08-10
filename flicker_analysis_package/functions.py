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
def make_SSVEPs(data, triggers, period):
    import numpy as np

    segment_matrix = np.zeros([len(triggers), period]) # empty matrix to put segments into
    
    seg_count = 0 # keep track of the number of segments
    
    # loop through all triggers and put the corresponding segment of data into the matrix
    for trigger in triggers:
        
        # select a segment of data the lenght of the flicker period, starting from the trigger time 
        segment =  data[trigger:trigger+period] 
        
        segment_matrix[seg_count,:] = segment

        seg_count += 1
    
    SSVEP = segment_matrix.mean(axis=0) # average to make SSVEP
    
    SSVEP = SSVEP - SSVEP.mean() # baseline correct
    
    return SSVEP




### signal to noise ratio by randomly shuffling the data points of each segment and then making the SSVEP, compare to true SSVEP
## instead of peak to peak amplitude, use a time range around the peak

def SNR_random(data, triggers, period):
    
    import numpy as np
    import random
    import matplotlib.pyplot as plt
    
    segment_matrix = np.zeros([len(triggers), period]) # empty matrix to put segments into
    random_segment_matrix = np.zeros([len(triggers), period]) # empty matrix to put the randomly shuffled segments into
    seg_count = 0 # keep track of the number of segments
    
    # loop through all triggers and put the corresponding segment of data into the matrix
    for trigger in triggers:
        
        # select a segment of data the lenght of the flicker period, starting from the trigger time 
        segment =  data[trigger:trigger+period] 
        segment_matrix[seg_count,:] = segment
        
        random.shuffle(segment)

        random_segment_matrix[seg_count,:] = segment

        seg_count += 1
    
    true_SSVEP = segment_matrix.mean(axis=0) # average to make SSVEP
    random_SSVEP = random_segment_matrix.mean(axis=0) # average to make SSVEP of the randomly shuffled data

    true_SSVEP = true_SSVEP - true_SSVEP.mean() # baseline correct
    random_SSVEP = random_SSVEP - random_SSVEP.mean()
   # plt.plot(random_SSVEP, '--k')

    for condition in ('true', 'random'):
        
        if condition == 'true':
            SSVEP = np.copy(true_SSVEP)
        elif condition == 'random':
            SSVEP = np.copy(random_SSVEP)

        SSVEP_repeated = np.tile(SSVEP, 2) # repeat the SSVEP in case the peak area is too near the beginning or end, loops around to the start  
    
        max_SSVEP_index = np.argmax(SSVEP) # get the index of the peak of the SSVEP
        
        ## get the average of the 5 data points around the peak of the SSVEP
        if (len(SSVEP) - max_SSVEP_index) <= 2: # if the peak is near the end of the SSVEP, 
            average_peak_area = SSVEP_repeated[max_SSVEP_index-2:max_SSVEP_index+3].mean()
        elif max_SSVEP_index <= 2: # if the peak index is near the begining of the SSVEP, repeat the SSVEP and move forward by the length of the SSVEP
            average_peak_area = SSVEP_repeated[max_SSVEP_index+len(SSVEP)-2:max_SSVEP_index+len(SSVEP)+3].mean()
        else: # otherwise, just average the area around the peak
            average_peak_area = SSVEP[max_SSVEP_index-2:max_SSVEP_index+3].mean()
        
        min_SSVEP_index = np.argmin(SSVEP) # get the index of the trough of the SSVEP
        
        ## get the average of the 5 data points around the trough of the SSVEP
        if (len(SSVEP) - min_SSVEP_index) <= 2: # if the trough is near the end of the SSVEP, 
            average_trough_area = SSVEP_repeated[min_SSVEP_index-2:min_SSVEP_index+3].mean()
        elif min_SSVEP_index <= 2: # if the trough index is near the begining of the SSVEP, move forward by the length of the SSVEP
            average_trough_area = SSVEP_repeated[min_SSVEP_index+len(SSVEP)-2:min_SSVEP_index+len(SSVEP)+3].mean()
        else: # otherwise, just average the area around the trough
            average_trough_area = SSVEP[min_SSVEP_index-2:min_SSVEP_index+3].mean()
    
    
        SSVEP_range = np.abs(average_peak_area - average_trough_area)

        if condition == 'true':
            true_SSVEP_range = np.copy(SSVEP_range)
        elif condition == 'random':
            random_SSVEP_range = np.copy(SSVEP_range)
    
    SNR = true_SSVEP_range/random_SSVEP_range
    
    return SNR



## Randomly split the triggers and create two SSVEPs, return the amplitude difference between the two 
def SSVEP_split_amplitude_difference(data, triggers, period):
    
    import numpy as np
    import random
 #   import matplotlib.pyplot as plt

    seg_nums = np.arange(0,len(triggers)) # an index for seach segment
 
    random.shuffle(seg_nums) # randomize the order
    
    for random_half in range(0,2): # select the first half, and then the second half, of the randomized segments, and make an SSVEP of each

        if random_half == 0:
            random_half_triggers = triggers[seg_nums[0:int(len(triggers)/2)]]
        elif random_half == 1:
            random_half_triggers = triggers[seg_nums[int(len(triggers)/2):]]

        segment_matrix = np.zeros([len(random_half_triggers), period]) # empty matrix to put the segments into
        seg_count = 0 # keep track of the number of segments
   
        for trigger in random_half_triggers:
            segment =  data[trigger:trigger+period] 
            segment_matrix[seg_count,:] = segment
            seg_count += 1

        SSVEP = segment_matrix[0:seg_count,:].mean(axis=0) # average to make SSVEP
        
        SSVEP = SSVEP - SSVEP.mean() # baseline correct

        if random_half == 0:
            SSVEP_1 = np.copy(SSVEP)
        elif random_half == 1:
            SSVEP_2 = np.copy(SSVEP)
   
    amplitude_difference = np.ptp(SSVEP_1) - np.ptp(SSVEP_2)

    return amplitude_difference

##  Randomly split the triggers from one condition to create two SSVEPs and return the correlation between the two
def compare_SSVEPs_split(data, triggers, period):
    
    import numpy as np
    import random
 #   import matplotlib.pyplot as plt

    seg_nums = np.arange(0,len(triggers)) # an index for seach segment
 
    random.shuffle(seg_nums) # randomize the order
    
    for random_half in range(0,2): # select the first half, and then the second half, of the randomized segments, and make an SSVEP of each

        if random_half == 0:
            random_half_triggers = triggers[seg_nums[0:int(len(triggers)/2)]]
        elif random_half == 1:
            random_half_triggers = triggers[seg_nums[int(len(triggers)/2):]]

        segment_matrix = np.zeros([len(random_half_triggers), period]) # empty matrix to put the segments into
        seg_count = 0 # keep track of the number of segments
   
        for trigger in random_half_triggers:
            segment =  data[trigger:trigger+period] 
            segment_matrix[seg_count,:] = segment
            seg_count += 1

        SSVEP = segment_matrix[0:seg_count,:].mean(axis=0) # average to make SSVEP
        
        SSVEP = SSVEP - SSVEP.mean() # baseline correct

        if random_half == 0:
            SSVEP_1 = np.copy(SSVEP)
        elif random_half == 1:
            SSVEP_2 = np.copy(SSVEP)
   
    correlation = np.corrcoef(SSVEP_1, SSVEP_2)[0,1]

    return correlation
    


##  Randomly split the triggers from one condition to create two SSVEPs and return directional phase shift between the two
def phase_shift_SSVEPs_split(data, triggers, period):
    
    import numpy as np
    import random
 #   import matplotlib.pyplot as plt

    seg_nums = np.arange(0,len(triggers)) # an index for seach segment
 
    random.shuffle(seg_nums) # randomize the order
    
    for random_half in range(0,2): # select the first half, and then the second half, of the randomized segments, and make an SSVEP of each

        if random_half == 0:
            random_half_triggers = triggers[seg_nums[0:int(len(triggers)/2)]]
        elif random_half == 1:
            random_half_triggers = triggers[seg_nums[int(len(triggers)/2):]]

        segment_matrix = np.zeros([len(random_half_triggers), period]) # empty matrix to put the segments into
        seg_count = 0 # keep track of the number of segments
   
        for trigger in random_half_triggers:
            segment =  data[trigger:trigger+period] 
            segment_matrix[seg_count,:] = segment
            seg_count += 1

        SSVEP = segment_matrix[0:seg_count,:].mean(axis=0) # average to make SSVEP
        
        SSVEP = SSVEP - SSVEP.mean() # baseline correct

        if random_half == 0:
            SSVEP_1 = np.copy(SSVEP)
        elif random_half == 1:
            SSVEP_2 = np.copy(SSVEP)
   
    phase_shift = cross_correlation_directional(SSVEP_1, SSVEP_2)

    return phase_shift

    
    # plt.legend()


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
def make_SSVEPs_random(data, triggers, period, num_loops):
    ## make SSVEP with all segments
    
    import numpy as np
    import matplotlib.pyplot as plt
    import random 

    segment_matrix = np.zeros([len(triggers), period]) # empty matrix to put segments into
    
    seg_count = 0 # keep track of the number of segments
    
    # loop through all triggers and put the corresponding segment of data into the matrix
    for trigger in triggers:
        
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
        
        shuffled_segment_matrix =  np.zeros([len(triggers), period])  
        
        # loop through all triggers and put the corresponding segment of data into the matrix
        seg_count = 0 # keep track of the number of segments
        
        for trigger in triggers:
            
            segment =  data[trigger:trigger+period] 
        
            random.shuffle(segment) # randomly shuffle the data points
            
            shuffled_segment_matrix[seg_count,:] = segment
            
            seg_count += 1
        
        random_SSVEP = shuffled_segment_matrix.mean(axis=0) # average to make SSVEP
        
        random_SSVEP = random_SSVEP - random_SSVEP.mean() # baseline correct
        
        random_amplitudes[loop] = np.ptp(random_SSVEP)
    
    
    # plt.plot(random_SSVEP,'k') # plot the last random shuffle, just to see
    
    # plt.plot(SSVEP,'b') # plot the true SSVEP
    
    
    
    true_amplitude = np.ptp(SSVEP)
    
    print('True amplitude = ', true_amplitude)
    
    average_noise = random_amplitudes.mean()
    
    print('Amplitude of noise = ', average_noise)
    
    std_noise = np.std(random_amplitudes)
    
    print('Standard Deviation noise = ', std_noise)
    
    Z_score  = (true_amplitude-average_noise) / std_noise
    
    print('Z_score = ', Z_score)
    
    return Z_score
    
##Function for only random SSVEPs and z score, with offset. Not necessary once trigger artefact is removed
def randomSSVEPs_zscore(SSVEP, data, all_triggers, period, num_loops, offset):
    
    import numpy as np
  #  import matplotlib.pyplot as plt
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
    
    
  #  plt.plot(random_SSVEP,'k') # plot the last random shuffle, just to see
    
 #   plt.plot(SSVEP,'b') # plot the true SSVEP
    
    true_amplitude = np.ptp(SSVEP)
    
    print('True amplitude = ', true_amplitude)
    
    average_noise = random_amplitudes.mean()
    
    print('Amplitude noise = ', average_noise)
    
    std_noise = np.std(random_amplitudes)
    
    print('Standard Deviation noise = ', std_noise)
    
    Z_score  = (true_amplitude-average_noise) / std_noise
    
    print('Z_score = ', Z_score)



##Function for linear interpolation of trigger artefacts (plot SSVEP showing before and after in this function)
## this function takes the trigger artefact times relative to the trigger and does the linear interpolation on the raw data, and returns a new data time series 

def linear_interpolation(data, triggers, time_1, time_2, trig_length):
    
    import numpy as np
    
    for trigger in triggers:

        data[trigger+time_1:trigger+time_1+trig_length+1] = np.linspace(data[trigger+time_1], data[trigger+time_1+trig_length], num = trig_length+1)
        data[trigger+time_2:trigger+time_2+trig_length+1] = np.linspace(data[trigger+time_2], data[trigger+time_2+trig_length], num = trig_length+1)
    
    
    return data


##Function for making induced FFT
#Desc: Segment data into segments of a given length, do an FFT on each segment and then average the FFTs.



def induced_fft(data, triggers, length, sample_rate): # length = length of segment to use in seconds (1/length = the frequency resolution), sample rate in Hz
    
    import numpy as np
    from scipy import fftpack
    
    length_of_segment = int(length * sample_rate)
    
    segment_matrix = np.zeros([len(triggers), length_of_segment]) # empty matrix to put segments into
    
    seg_count = 0
   
    k = 0
    
    while k < len(data) - length_of_segment: # loop until the end of data
    
        if k in triggers: # if data point is a trigger
        
            segment = data[k:k+length_of_segment] # get a segment of data
    
            segment = segment - segment.mean() # baseline correct
                
            segment_hanning = segment * np.hanning(length_of_segment) # multiply by hanning window
                
            fft_segment = np.abs(fftpack.fft(segment_hanning)) # FFT
    
            segment_matrix[seg_count,:] = fft_segment # put into matrix
    
            seg_count+=1
            
            k = k + length_of_segment # move forward the length of the segment, so segments are not overlapping
    
        k+=1
    
    fft_spectrum = segment_matrix[0:seg_count,:].mean(axis=0)
    
    return fft_spectrum



##Function for making evoked FFT
#Desc: Segment data into non-overlapping segments of a given length, each time locked to a trigger. Then average and do an FFT on the average.


def evoked_fft(data, triggers, length, sample_rate): # length = length of segment to use in seconds (1/length = the frequency resolution), sample rate in Hz
    
    import numpy as np
    from scipy import fftpack
    
    length_of_segment = int(length * sample_rate)
    
    segment_matrix = np.zeros([len(triggers), length_of_segment]) # empty matrix to put segments into
    
    seg_count = 0
   
    k = 0
    
    while k < len(data) - length_of_segment: # loop until the end of data
    
        if k in triggers: # if data point is a trigger
        
            segment = data[k:k+length_of_segment] # get a segment of data
    
            segment_matrix[seg_count,:] = segment # put into matrix
    
            seg_count+=1
            
            k = k + length_of_segment # move forward the length of the segment, so segments are not overlapping
    
        k+=1
    
    SSVEP = segment_matrix[0:seg_count,:].mean(axis=0)
    
    
    SSVEP = SSVEP - SSVEP.mean() # baseline correct
        
    SSVEP_hanning = SSVEP * np.hanning(length_of_segment) # multiply by hanning window
        
    fft_SSVEP = np.abs(fftpack.fft(SSVEP_hanning)) # FFT

    return fft_SSVEP
            

###ANALYSIS FUNCTIONS that require making SSSVEPs

##Permutation tests on two conditions
#Desc: Make two SSVEPs by randomly assigning segments from condition 1 and condition 2 to two make two groups, and compare the difference in amplitude. Repeat this may times (e.g. 1000) and create a distribution of amplitude differences, which can be compared to the true difference.

##Permutation tests on signal-to-noise
#Desc: Make SSVEPs using a random subset of triggers (start with a small number), repeat many times (e.g.1000). Then repeat entire test with a different number of segments. Progressively increase number of segments and keep track of the resulting SSVEP.

##Simulated SSVEPs (for data without flicker)
#Desc: Add a simulated SSVEP (e.g. a sine wave) to the data and create a numpy array of “trigger times”, such that the data can be used with the above functions.

##Plots (for plotting example SSVEPs)


##### ANALYSIS FUNCTIONS ON SSVEPs that are already averaged

def time_warp_SSVEPs(SSVEP_1, SSVEP_2):
    
    import numpy as np
    

    #amplitude_difference = np.ptp(SSVEP_1) - np.ptp(SSVEP_2)

    # check if SSVEPs are the same length
    
    if len(SSVEP_1) == len(SSVEP_2):
        
        print('SSVEPs are already the same length')
        
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
        
        for k in range(0,len(shorter_SSVEP)-1): # for every time point in the longer SSVEP
            t = int(steps_to_use[k]) # get the data value of one time point
            t_2 = int(steps_to_use[k+1]) # and of the following time point
            temp_SSVEP[t:t_2] = np.linspace(shorter_SSVEP[k],shorter_SSVEP[k+1],len(temp_SSVEP[t:t_2])) # add these data points, and linear interpolation between them, to the temp SSVEP
           
        
        
        new_SSVEP = np.zeros([len(longer_SSVEP),]) # empty array to put the new values into, the length of the longer SSVEP
        
        
        steps_to_use = np.linspace(0,length_temp_SSVEP,len(longer_SSVEP)) # evenly spaced time points,  one for each time point of the longer SSVEP
        steps_to_use = np.round(steps_to_use) # round these time points to integer values
        
        for k in range(0,len(longer_SSVEP)-1): # get the values from the longer SSVEP, and use them to make the new SSVEP
            t = int(steps_to_use[k])   
            new_SSVEP[k] = temp_SSVEP[t]
        
        
        new_SSVEP[-1] = temp_SSVEP[length_temp_SSVEP-1] # put the last time point in seperatly


        correlation = np.corrcoef(longer_SSVEP, new_SSVEP)    # Pearson correlation of the longer SSVEP and the time warped shorter SSVEP

    
    return correlation
    return new_SSVEP

## the absolute phase shift
def cross_correlation_absolute(SSVEP_1, SSVEP_2):
    
    import numpy as np
    
    if len(SSVEP_1) == len(SSVEP_2): # check the two SSVEPs are the same length
     
        correlations = np.zeros([len(SSVEP_1),]) # empty array to put the correlation values into
        
        phase_shifted_SSVEP_2 = np.copy(SSVEP_2) # make a copy of SSVEP_2 which can be phase shifted
        
        for i in range(len(SSVEP_1)):
        
           # plt.plot(phase_shift_SSVEP_2)
        
            correlations[i] = np.corrcoef(SSVEP_1, phase_shifted_SSVEP_2)[1,0] # get the pearson's correlatios for this phase value
        
            phase_shifted_SSVEP_2 = np.roll(phase_shifted_SSVEP_2, 1) # phase shift by one data point
            
           
        # get the phase shift 
        
        phase_lag = np.argmax(correlations)  # the phase lag with the maximum correlation value

        #print('phase lag = ' + str(phase_lag))
        
    else:
        
        print('SSVEPs are not the same length')
        
        phase_lag = 0
            
        
    phase_lag_degrees = phase_lag/len(SSVEP_1) * 360

    #print('Phase lag in degrees = ' + str(phase_lag_degrees))
        
    ## convert to absolute phase lag, i.e. independent of direction, max value will be 180
    if phase_lag_degrees > 180:
        absolute_phase_lag_degrees = 360-phase_lag_degrees
    else:
        absolute_phase_lag_degrees = phase_lag_degrees


    return absolute_phase_lag_degrees

#### direction specific phase shift - if phase shift is greater than 180, then assume the phase shift is in the other direction, i.e. negative
def cross_correlation_directional(SSVEP_1, SSVEP_2):
    
    import numpy as np
    
    if len(SSVEP_1) == len(SSVEP_2): # check the two SSVEPs are the same length
     
        correlations = np.zeros([len(SSVEP_1),]) # empty array to put the correlation values into
        
        phase_shifted_SSVEP_2 = np.copy(SSVEP_2) # make a copy of SSVEP_2 which can be phase shifted
        
        for i in range(len(SSVEP_1)):
        
           # plt.plot(phase_shift_SSVEP_2)
        
            correlations[i] = np.corrcoef(SSVEP_1, phase_shifted_SSVEP_2)[1,0] # get the pearson's correlatios for this phase value
        
            phase_shifted_SSVEP_2 = np.roll(phase_shifted_SSVEP_2, 1) # phase shift by one data point
            
           
        # get the phase shift 
        
        phase_lag = np.argmax(correlations)  # the phase lag with the maximum correlation value

        #print('phase lag = ' + str(phase_lag))
        
    else:
        
        print('SSVEPs are not the same length')
        
        phase_lag = 0
            
        
    phase_lag_degrees = phase_lag/len(SSVEP_1) * 360

    #print('Phase lag in degrees = ' + str(phase_lag_degrees))
        
    ## convert to directional phase lag, i.e. if the phase lag ig greater than 180 degrees, assume in the other direction and negative
    if phase_lag_degrees > 180:
        directional_phase_lag_degrees = (360 - phase_lag_degrees) * -1
    else:
        directional_phase_lag_degrees = phase_lag_degrees


    return directional_phase_lag_degrees




######### general stats   ##################


### group level permutation test, within subjects shuffle


def group_permutation_test(scores_condition_1, scores_condition_2):
    
    import numpy as np
    from random import choice
    
    ## check for Nans and remove them from both sets of data
    if len(np.argwhere(np.isnan(scores_condition_1))) > 0:
        
        index = np.argwhere(np.isnan(scores_condition_1))
        scores_condition_1 = np.delete(scores_condition_1, index)
        scores_condition_2 = np.delete(scores_condition_2, index)
        
    if len(np.argwhere(np.isnan(scores_condition_2))) > 0:

        index = np.argwhere(np.isnan(scores_condition_2))
        scores_condition_1 = np.delete(scores_condition_1, index)
        scores_condition_2 = np.delete(scores_condition_2, index)
        
        
    
    ## check groups are the same size
    
    if len(scores_condition_1) == len(scores_condition_2):
    
        num_subjects = len(scores_condition_1)

    else: # if groups are not the same size, take the length of the smallest group and the same number of subjects from the larger group
    
        print('ERROR: Variables are not the same length. Resizing ...')
    
        if len(scores_condition_1) > len(scores_condition_2):
            
            scores_condition_1 = scores_condition_1[0:len(scores_condition_2)]
            
            num_subjects = len(scores_condition_2)
            
        elif len(scores_condition_1) < len(scores_condition_2):
            
            scores_condition_2 = scores_condition_2[0:len(scores_condition_1)]
    
            num_subjects = len(scores_condition_1)
    


    average_condition_1 = scores_condition_1.mean()
    
    average_condition_2 = scores_condition_2.mean()
    
    true_difference = average_condition_1 - average_condition_2
    
    num_loops = 1000
    
    shuffled_differences = np.zeros([num_loops,])
    
    for loop in range(0,num_loops):
        
        random_condition_1 = np.zeros([len(scores_condition_1),])
        random_condition_2 = np.zeros([len(scores_condition_2),])
        
        for subject in range(0,num_subjects):
            
            decide = choice(['yes', 'no'])  # for each subject, decide to either keep the correct labels, or swap the conditions. 50% chance
                
            if decide == 'yes':
                
                random_condition_1[subject] = scores_condition_1[subject] # keep the correct labels
                random_condition_2[subject] = scores_condition_2[subject]
            
            elif decide == 'no':
                
                random_condition_1[subject] = scores_condition_2[subject] # swap the labels
                random_condition_2[subject] = scores_condition_1[subject]
            
        average_random_condition_1 = random_condition_1.mean()
        average_random_condition_2 = random_condition_2.mean()

        difference_random_conditions = average_random_condition_1 - average_random_condition_2

        shuffled_differences[loop] = difference_random_conditions
        
        
    
    Z_score = (true_difference - shuffled_differences.mean()) / np.std(shuffled_differences) # calculate Z score

    return Z_score



# calculate the Cohen's d between two samples
def cohens_d(scores_condition_1, scores_condition_2):
    
    import numpy as np
    from numpy import var

    ## check for Nans and remove them from both sets of data
    if len(np.argwhere(np.isnan(scores_condition_1))) > 0:
        
        index = np.argwhere(np.isnan(scores_condition_1))
        scores_condition_1 = np.delete(scores_condition_1, index)
        scores_condition_2 = np.delete(scores_condition_2, index)
        
    if len(np.argwhere(np.isnan(scores_condition_2))) > 0:

        index = np.argwhere(np.isnan(scores_condition_2))
        scores_condition_1 = np.delete(scores_condition_1, index)
        scores_condition_2 = np.delete(scores_condition_2, index)



    num_subjects_1 = len(scores_condition_1)
    num_subjects_2 = len(scores_condition_2)
    
    # calculate the variance of the samples
    s1, s2 = var(scores_condition_1, ddof=1), var(scores_condition_2, ddof=1) 
 	# calculate the pooled standard deviation
    s = np.sqrt(((num_subjects_1 - 1) * s1 + (num_subjects_2 - 1) * s2) / (num_subjects_1 + num_subjects_2 - 2))
 	# calculate the means of the samples
    u1, u2 = scores_condition_1.mean(), scores_condition_2.mean()
 	# calculate the effect size
    cohens_d = (u1 - u2) / s

    return cohens_d


# find largest cluster of Z scores for cluster based permutation tests. Requires adjacency matrix from MNE
# sig_cutoff = the significance cutoff value for inclusion in the clusters

def find_max_cluster(Z_scores, adjacency, ch_names, sig_cutoff):
    
        
    significant_electrodes = []
    for electrode in range(0,64):
        if Z_scores[electrode] > sig_cutoff:
            significant_electrodes.append(electrode)
     
    all_clusters_summed_Z_scores = []
    
    for electrode in significant_electrodes:
    
        current_cluster = [] # for each significant electrode create a running list of the cluster
        
        neighbours_to_check = [] # # list to keep track of neighbours which still need to be checked for further significant neighbours
        
        # print('  ')      
        # print('Electrode ' + ch_names[electrode] + '  significant cluster:')    
        # print('  ')   
        
        # check all significant electrodes to see if they are an immediate neighbour
        for electrode_to_check in significant_electrodes: 
            if adjacency[electrode,electrode_to_check] == 1:
                
               # print(ch_names[electrode_to_check])    
                # add significant neighbours to the cluster list and the the neighbours-to-checked list 
                current_cluster.append(electrode_to_check)
                neighbours_to_check.append(electrode_to_check)
        
        # remove the current electrode from the neighbours-to-be-checked list, it's already been checked
        neighbours_to_check.remove(electrode) 
        
        # check all neighbours for significant neighbours, removing them from the list when checked
        while len(neighbours_to_check) > 0: 
            
            checking_current_neighbour = neighbours_to_check[0] # take the first neighbour from the list
            
            # check all significant electrodes to see if they are a neighbour
            for electrode_to_check in significant_electrodes:
                if adjacency[checking_current_neighbour,electrode_to_check] == 1:
                    if electrode_to_check not in current_cluster: # only include electrodes not already in the cluster
                        # add significant neighbours to the cluster list and the the neighbours-to-checked list 
                        current_cluster.append(electrode_to_check)
                        neighbours_to_check.append(electrode_to_check)
                      #  print(ch_names[electrode_to_check]) 
            
            neighbours_to_check.pop(0) # remove the first neighbour from the list
            
        
        cluster_Z_scores = []
        
        for electrode in current_cluster:
           # print(electrode)
            cluster_Z_scores.append(Z_scores[electrode])
        
        summed_Z_scores = sum(cluster_Z_scores)
        
        all_clusters_summed_Z_scores.append(summed_Z_scores)
    
    max_cluster = max(all_clusters_summed_Z_scores)
        
    return max_cluster   