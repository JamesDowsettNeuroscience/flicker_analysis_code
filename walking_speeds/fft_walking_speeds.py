#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 09:36:48 2021

@author: James Dowsett
"""

import numpy as np

from scipy import signal, fftpack

import matplotlib.pyplot as plt

## load data

subject = 1

file_name = 'pilot_' + str(subject)

all_data = np.load(file_name + '_all_data.npy')


electrode = 10 # channel 10 is the vertical axis of the accelerometer

data = all_data[electrode,:]


## load triggers
all_triggers = np.load(file_name + '_all_triggers.npy')

# each trial begins with 5 beep sounds, with triggers, at the right frequency. 
# Define the period, i.e. the time between triggers
period_slow = 1000

period_mid = 666

period_fast = 500


diff_all_triggers = np.diff(all_triggers) # first differential, to see the time between triggers




## empty matricies to put FFTs into

length_of_segment = 10000 # 10 second segments of data

slow_fft_matrix = np.zeros([100,length_of_segment])
slow_seg_count = 0 

mid_fft_matrix = np.zeros([100,length_of_segment])
mid_seg_count = 0 

fast_fft_matrix = np.zeros([100,length_of_segment])
fast_seg_count = 0 


trigger_time_series = np.zeros([len(data),]) # time series to plot the triggers onto the data, to check the timing

## loop for all triggers
trigger = 0
while trigger < len(all_triggers)-10:
    
    # work out the frequency of the beeps by checking the time between triggers
    
    next_5_triggers = all_triggers[trigger:trigger+5]
    difference_time_between_triggers = np.diff(next_5_triggers)
    
    #time_to_next_trigger = all_triggers[trigger+1] - all_triggers[trigger]
     
    # check if the average time between triggers is sufficently close to the walking period.
    # this should exclude any random triggers that may have occured, which are not part of a consistent train of 5
    if np.abs(difference_time_between_triggers.mean() - period_slow) < 2:
        walking_frequency = 1
        trigger_time_series[all_triggers[trigger]] = 1000
             
    elif np.abs(difference_time_between_triggers.mean() - period_mid) < 2:
        walking_frequency = 1.5
        trigger_time_series[all_triggers[trigger]] = 666
       
    elif np.abs(difference_time_between_triggers.mean() - period_fast) < 2:
        walking_frequency = 2
        trigger_time_series[all_triggers[trigger]] = 500
       
    else:
        walking_frequency = 0
        trigger +=1
        
    if walking_frequency > 0:    # only continue if triggr is the first of 5 evenly spaced triggers at one of the frequencies
        
        walking_start_time = all_triggers[trigger+4] # start from the last trigger
        
        
        segment = data[walking_start_time:walking_start_time+length_of_segment]
        
        segment = segment - segment.mean() # baseline correct
             
        segment = segment * np.hanning(len(segment)) # multiply by hanning window
         
        fft_segment = np.abs(fftpack.fft(segment)) # FFT
        
        if walking_frequency == 1:
            slow_fft_matrix[slow_seg_count,:] = fft_segment
            slow_seg_count += 1
        elif walking_frequency == 1.5:
            mid_fft_matrix[mid_seg_count,:] = fft_segment
            mid_seg_count += 1
        elif walking_frequency == 2:
            fast_fft_matrix[fast_seg_count,:] = fft_segment
            fast_seg_count += 1    
        
        trigger = trigger + 4
    
print(str(slow_seg_count) + ' slow walking trials')
print(str(mid_seg_count) + ' mid walking trials')
print(str(fast_seg_count) + ' fast walking trials')  
    
    
    
average_fft_slow = slow_fft_matrix[0:slow_seg_count,:].mean(axis = 0)
average_fft_mid = mid_fft_matrix[0:mid_seg_count,:].mean(axis = 0)
average_fft_fast = fast_fft_matrix[0:fast_seg_count,:].mean(axis = 0)
    
sample_rate = 1000
  
time_axis = np.arange(0,length_of_segment,sample_rate/length_of_segment) # time_vector to plot FFT  
  
plt.figure()

plt.suptitle('Subject ' + str(subject))

# plot slow
plt.subplot(1,3,1)
    
plt.plot(time_axis[0:len(average_fft_slow)],average_fft_slow)

plt.plot(np.zeros([int(max(average_fft_slow)),]) + 1, np.arange(0, int(max(average_fft_slow))), '--k')    # dotted black line to show correct frequency
   
plt.title('slow')

plt.xlim([0, 5])


## plot mid

plt.subplot(1,3,2)
    
plt.plot(time_axis[0:len(average_fft_mid)],average_fft_mid)

plt.plot(np.zeros([int(max(average_fft_mid)),]) + 1.5, np.arange(0, int(max(average_fft_mid))), '--k')    # dotted black line to show correct frequency
   

plt.title('Mid')

plt.xlim([0, 5])




## plot fast

plt.subplot(1,3,3)
    
plt.plot(time_axis[0:len(average_fft_fast)],average_fft_fast)

plt.plot(np.zeros([int(max(average_fft_fast)),]) + 2, np.arange(0, int(max(average_fft_fast))), '--k')    # dotted black line to show correct frequency
   

plt.title('Fast')

plt.xlim([0, 5])



