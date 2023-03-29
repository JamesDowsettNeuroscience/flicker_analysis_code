#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 12:02:57 2021

@author: prithasen
"""




###COMMON ANALYSIS FUNCTIONS

##Function for high pass filter
def high_pass_filter(cutoff_frequency, sample_rate, data):
    
    from scipy import signal
    
    high_pass_filter = signal.butter(2, cutoff_frequency, 'hp', fs=sample_rate, output='sos')
    high_pass_filtered_data = signal.sosfilt(high_pass_filter, data)
    return high_pass_filtered_data

        
##Function for low pass filter 
def low_pass_filter(cutoff_frequency, sample_rate, data):
    
    from scipy import signal
    
    low_pass_filter = signal.butter(2, cutoff_frequency, 'lp', fs=sample_rate, output='sos')
    low_pass_filtered_data = signal.sosfilt(low_pass_filter, data)
    return low_pass_filtered_data
    

###ANALYSIS FUNCTIONS that require making SSSVEPs


## Decoding by correlation. Function takes two conditions: two arrays of data and corresponding triggers.
## For each loop the function randomly selects half of the triggers for each condition to create "training SSVEPs"
## With the remaining triggers, two "test SSVEPs" are created
## A correlation for each "test SSVEP" is done with each "training SSVEP"; if the correlation with the correct SSVEP is higher,
## it is awarded a score of one, if not it scores zero.
## This procedure is repeated [num_loops] times and the percent correct answers is returned

def decode_correlation(data_1, data_2, triggers_1, triggers_2, period, num_loops):
    
    import numpy as np
    import random
    # import matplotlib.pyplot as plt
    
    scores_condition_1 = np.zeros([num_loops,])
    scores_condition_2 = np.zeros([num_loops,])
    
    for loop in range(0,num_loops):     

        ## split triggers into training and test with a 50/50 split
        
        for condition in (1,2):

            if condition == 1:
                num_triggers = len(triggers_1)
                seg_nums = np.arange(0,num_triggers) # an index for each segment
            elif condition == 2:
                num_triggers = len(triggers_2)
                seg_nums = np.arange(0,num_triggers) 
                
         
            random.shuffle(seg_nums) # randomize the order
            
            # get half of the triggers
            training_trig_nums = seg_nums[0:int(num_triggers/2)] # first half
            test_trig_nums = seg_nums[int(num_triggers/2):num_triggers] # second half
            
            # get the corresponding trigger times
            if condition == 1:
                training_triggers_condition_1 = triggers_1[training_trig_nums]  
                test_triggers_condition_1 = triggers_1[test_trig_nums]
            elif condition == 2:
                training_triggers_condition_2 = triggers_2[training_trig_nums]
                test_triggers_condition_2 = triggers_2[test_trig_nums]



        ### make training SSVEPs
        training_SSVEP_condition_1 = make_SSVEPs(data_1, training_triggers_condition_1, period) 
        training_SSVEP_condition_2 = make_SSVEPs(data_2, training_triggers_condition_2, period) 

        # plt.plot(training_SSVEP_condition_1,'r')
        # plt.plot(training_SSVEP_condition_2,'b')
    
                
        ####  make test SSVEPs
        test_SSVEP_condition_1 = make_SSVEPs(data_1, test_triggers_condition_1, period) 
        test_SSVEP_condition_2 = make_SSVEPs(data_2, test_triggers_condition_2, period) 

    
       # plt.plot(test_SSVEP_condition_1,'m')
       # plt.plot(test_SSVEP_condition_2,'c')

        ## test condition 1 decoding
        corr_condition_1_test_and_condition_1_training = np.corrcoef(test_SSVEP_condition_1,training_SSVEP_condition_1)[0,1]
        corr_condition_1_test_and_condition_2_training = np.corrcoef(test_SSVEP_condition_1,training_SSVEP_condition_2)[0,1]

        if corr_condition_1_test_and_condition_1_training > corr_condition_1_test_and_condition_2_training:
            scores_condition_1[loop] = 1 # score one point
           

        ## test condition 2 decoding
        corr_condition_2_test_and_condition_1_training = np.corrcoef(test_SSVEP_condition_2,training_SSVEP_condition_1)[0,1]
        corr_condition_2_test_and_condition_2_training = np.corrcoef(test_SSVEP_condition_2,training_SSVEP_condition_2)[0,1]
       
        if corr_condition_2_test_and_condition_2_training > corr_condition_2_test_and_condition_1_training:
            scores_condition_2[loop] = 1 # score one point
                        
            
                
    percent_correct_condition_1 = np.sum(scores_condition_1) * (100/num_loops)
    percent_correct_condition_2 = np.sum(scores_condition_2) * (100/num_loops)
       
    average_percent_correct = (percent_correct_condition_1 + percent_correct_condition_2) / 2
    
    return average_percent_correct 
    
    

## the same decoding as above but with three conditions

def decode_correlation_3way(data_1, data_2, data_3, triggers_1, triggers_2, triggers_3, period, num_loops):
    
    import numpy as np
    import random
    # import matplotlib.pyplot as plt
    
    scores = np.zeros([num_loops,3])
   
    
    for loop in range(0,num_loops):     

        ## split triggers into training and test with a 50/50 split
        
        for condition in (1,2,3):

            if condition == 1:
                num_triggers = len(triggers_1)
                seg_nums = np.arange(0,num_triggers) # an index for each segment
            elif condition == 2:
                num_triggers = len(triggers_2)
                seg_nums = np.arange(0,num_triggers) 
            elif condition == 3:
                num_triggers = len(triggers_3)
                seg_nums = np.arange(0,num_triggers) 
                
         
            random.shuffle(seg_nums) # randomize the order
            
            # get half of the triggers
            training_trig_nums = seg_nums[0:int(num_triggers/2)] # first half
            test_trig_nums = seg_nums[int(num_triggers/2):num_triggers] # second half
            
            # get the corresponding trigger times
            if condition == 1:
                training_triggers_condition_1 = triggers_1[training_trig_nums]  
                test_triggers_condition_1 = triggers_1[test_trig_nums]
            elif condition == 2:
                training_triggers_condition_2 = triggers_2[training_trig_nums]
                test_triggers_condition_2 = triggers_2[test_trig_nums]
            elif condition == 3:
                training_triggers_condition_3 = triggers_3[training_trig_nums]
                test_triggers_condition_3 = triggers_3[test_trig_nums]            



        ### make training SSVEPs
        training_SSVEP_condition_1 = make_SSVEPs(data_1, training_triggers_condition_1, period) 
        training_SSVEP_condition_2 = make_SSVEPs(data_2, training_triggers_condition_2, period) 
        training_SSVEP_condition_3 = make_SSVEPs(data_3, training_triggers_condition_3, period) 

        # plt.plot(training_SSVEP_condition_1,'b')
        # plt.plot(training_SSVEP_condition_2,'r')
        # plt.plot(training_SSVEP_condition_3,'g')
    
                
        ####  make test SSVEPs
        test_SSVEP_condition_1 = make_SSVEPs(data_1, test_triggers_condition_1, period) 
        test_SSVEP_condition_2 = make_SSVEPs(data_2, test_triggers_condition_2, period) 
        test_SSVEP_condition_3 = make_SSVEPs(data_3, test_triggers_condition_3, period)

    
        # plt.plot(test_SSVEP_condition_1,'c')
        # plt.plot(test_SSVEP_condition_2,'m')
        # plt.plot(test_SSVEP_condition_3,'k')
        

        ## test condition 1 decoding
        corr_condition_1_test_and_condition_1_training = np.corrcoef(test_SSVEP_condition_1,training_SSVEP_condition_1)[0,1]
        corr_condition_1_test_and_condition_2_training = np.corrcoef(test_SSVEP_condition_1,training_SSVEP_condition_2)[0,1]
        corr_condition_1_test_and_condition_3_training = np.corrcoef(test_SSVEP_condition_1,training_SSVEP_condition_3)[0,1]

        if corr_condition_1_test_and_condition_1_training > corr_condition_1_test_and_condition_2_training and\
            corr_condition_1_test_and_condition_1_training > corr_condition_1_test_and_condition_3_training:
            
            scores[loop,0] = 1 # score one point
           

        ## test condition 2 decoding
        corr_condition_2_test_and_condition_1_training = np.corrcoef(test_SSVEP_condition_2,training_SSVEP_condition_1)[0,1]
        corr_condition_2_test_and_condition_2_training = np.corrcoef(test_SSVEP_condition_2,training_SSVEP_condition_2)[0,1]
        corr_condition_2_test_and_condition_3_training = np.corrcoef(test_SSVEP_condition_2,training_SSVEP_condition_3)[0,1]
        
        if corr_condition_2_test_and_condition_2_training > corr_condition_2_test_and_condition_1_training and\
            corr_condition_2_test_and_condition_2_training > corr_condition_2_test_and_condition_3_training:
                
            scores[loop,1] = 1 # score one point
                        
        ## test condition 3 decoding
        corr_condition_3_test_and_condition_1_training = np.corrcoef(test_SSVEP_condition_3,training_SSVEP_condition_1)[0,1]
        corr_condition_3_test_and_condition_2_training = np.corrcoef(test_SSVEP_condition_3,training_SSVEP_condition_2)[0,1]
        corr_condition_3_test_and_condition_3_training = np.corrcoef(test_SSVEP_condition_3,training_SSVEP_condition_3)[0,1]
        
        if corr_condition_3_test_and_condition_3_training > corr_condition_3_test_and_condition_1_training and\
            corr_condition_3_test_and_condition_3_training > corr_condition_3_test_and_condition_2_training:
                
            scores[loop,2] = 1 # score one point
                                   
            
                
    percent_correct_condition_1 = np.sum(scores[:,0]) * (100/num_loops)
    percent_correct_condition_2 = np.sum(scores[:,1]) * (100/num_loops)
    percent_correct_condition_3 = np.sum(scores[:,2]) * (100/num_loops)
       
    average_percent_correct = (percent_correct_condition_1 + percent_correct_condition_2 + percent_correct_condition_3) / 3
    
    return average_percent_correct 
    
    










##Function to make SSVEPs (use triggers & average)
def make_SSVEPs(data, triggers, period):
    import numpy as np

    segment_matrix = np.zeros([len(triggers), period]) # empty matrix to put segments into
    
    seg_count = 0 # keep track of the number of segments
    
    # loop through all triggers and put the corresponding segment of data into the matrix
    for trigger in triggers:
        
        # select a segment of data the lenght of the flicker period, starting from the trigger time 
        segment =  data[trigger:trigger+period] 
        
        if len(segment) == period:
            segment_matrix[seg_count,:] = segment
    
            seg_count += 1
        

    SSVEP = segment_matrix[0:seg_count,:].mean(axis=0) # average to make SSVEP
    
    SSVEP = SSVEP - SSVEP.mean() # baseline correct
    
    
    return SSVEP


##Function to make SSVEPs but randomly shuffling the data in each segment
def make_SSVEPs_random(data, triggers, period):
    import numpy as np
    import random

    segment_matrix = np.zeros([len(triggers), period]) # empty matrix to put segments into
    
    seg_count = 0 # keep track of the number of segments
    
    # loop through all triggers and put the corresponding segment of data into the matrix
    for trigger in triggers:
        
        # select a segment of data the lenght of the flicker period, starting from the trigger time 
        segment =  data[trigger:trigger+period] 
        
        randomised_segment = np.copy(segment)
        
        random.shuffle(randomised_segment)
        
        if len(segment) == period:
            segment_matrix[seg_count,:] = randomised_segment
    
            seg_count += 1
        

    SSVEP = segment_matrix[seg_count,:].mean(axis=0) # average to make SSVEP
    
    SSVEP = SSVEP - SSVEP.mean() # baseline correct
    
    
    return SSVEP




### signal to noise ratio by randomly shuffling the data points of each segment and then making the SSVEP

def SNR_random(data, triggers, period):
    
    import numpy as np
    import random
    #import matplotlib.pyplot as plt
    
    segment_matrix = np.zeros([len(triggers), period]) # empty matrix to put segments into
    random_segment_matrix = np.zeros([len(triggers), period]) # empty matrix to put the randomly shuffled segments into
    seg_count = 0 # keep track of the number of segments
    
    # loop through all triggers and put the corresponding segment of data into the matrix
    for trigger in triggers:
        
        # select a segment of data the lenght of the flicker period, starting from the trigger time 
        segment = np.copy(data[trigger:trigger+period]) 
        segment_matrix[seg_count,:] = np.copy(segment)
        
        random.shuffle(segment)

        random_segment_matrix[seg_count,:] = np.copy(segment)

        seg_count += 1
    
    true_SSVEP = segment_matrix.mean(axis=0) # average to make SSVEP
    random_SSVEP = random_segment_matrix.mean(axis=0) # average to make SSVEP of the randomly shuffled data

    # true_SSVEP = true_SSVEP - true_SSVEP.mean() # baseline correct
    # random_SSVEP = random_SSVEP - random_SSVEP.mean()


    true_SSVEP_range = np.ptp(true_SSVEP )

    random_SSVEP_range = np.ptp(random_SSVEP)

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






##Function for linear interpolation of trigger artefacts (plot SSVEP showing before and after in this function)
## this function takes the trigger artefact times relative to the trigger and does the linear interpolation on the raw data, and returns a new data time series 

def linear_interpolation(data, triggers, time_1, time_2, trig_length):
    
    import numpy as np
    
    for trigger in triggers:

        data[trigger+time_1:trigger+time_1+trig_length+1] = np.linspace(data[trigger+time_1], data[trigger+time_1+trig_length], num = trig_length+1)
        data[trigger+time_2:trigger+time_2+trig_length+1] = np.linspace(data[trigger+time_2], data[trigger+time_2+trig_length], num = trig_length+1)
    
    
    return data





##Function for making induced FFT
#Description: Segment data into segments of a given length, do an FFT on each segment and then average the FFTs.


def induced_fft(data, length, sample_rate): # length = length of segment to use in seconds (1/length = the frequency resolution), sample rate in Hz
    
    import numpy as np
    from scipy import fftpack
    
    length_of_segment = int(length * sample_rate)
    
    estimated_num_segs = int((len(data)/length_of_segment) + 1)
    
    segment_matrix = np.zeros([estimated_num_segs, length_of_segment]) # empty matrix to put segments into
    
    seg_count = 0
   
    k = 0
    
    while k < len(data) - length_of_segment: # loop until the end of data

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
            





##### ANALYSIS FUNCTIONS ON SSVEPs that are already averaged


# time warp SSVEPs of different lengths so they can be compared

def time_warp_SSVEPs(SSVEP_1, SSVEP_2):
    
    import numpy as np
    

    #amplitude_difference = np.ptp(SSVEP_1) - np.ptp(SSVEP_2)

    # check if SSVEPs are the same length
    
    if len(SSVEP_1) == len(SSVEP_2):
        
     #   print('SSVEPs are already the same length')
        new_SSVEP = np.copy(SSVEP_1)
       # correlation = np.corrcoef(SSVEP_1, SSVEP_2)    # Pearson correlation

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


       # correlation = np.corrcoef(longer_SSVEP, new_SSVEP)    # Pearson correlation of the longer SSVEP and the time warped shorter SSVEP

    
  #  return correlation
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


## calculate the max correlation after cross correlation
def max_correlation(SSVEP_1, SSVEP_2):
    
    import numpy as np
    
    correlations = np.zeros([len(SSVEP_1),]) # empty array to put the correlation values into
    
    
    if len(SSVEP_1) == len(SSVEP_2): # check the two SSVEPs are the same length

        phase_shifted_SSVEP_2 = np.copy(SSVEP_2) # make a copy of SSVEP_2 which can be phase shifted
        
        for i in range(len(SSVEP_1)):
        
           # plt.plot(phase_shift_SSVEP_2)
        
            correlations[i] = np.corrcoef(SSVEP_1, phase_shifted_SSVEP_2)[1,0] # get the pearson's correlatios for this phase value
        
            phase_shifted_SSVEP_2 = np.roll(phase_shifted_SSVEP_2, 1) # phase shift by one data point

    else:
        
        print('SSVEPs are not the same length')

    max_correlation = max(correlations)   

    return max_correlation


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
    
    if len(significant_electrodes) == 0:
        max_cluster = 0
    elif len(significant_electrodes) > 0:
        
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