#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 14:17:11 2021

@author: James Dowsett
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import time


## load data from one electrode and EOG data

subject = 2

file_name = 'S' + str(subject) + '_colour'

electrode_name = 'Oz'

electrode_data_file_name = file_name + '_' + electrode_name +'_data.npy'
VEOG_data_file_name = file_name + '_VEOG_data.npy'
HEOG_data_file_name = file_name + '_HEOG_data.npy'

data = np.load(electrode_data_file_name)
VEOG_data = np.load(VEOG_data_file_name)
HEOG_data = np.load(HEOG_data_file_name)

start_time = time.process_time() # keep track of how long the script takes in total 


plt.figure()

flicker_periods = [33, 28, 25, 22, 18, 17]


# shift the SSVEP by some small amount so the trigger is not at the edge of the segment
offset = 1 

# time when the flicker trigger will appear in the final SSVEP, for removal of the trigger artefact
trigger_2_times = [84, 69, 64, 54, 44, 44]


# empty matrices to store amplitude and correlation values in

colour_amplitude_values = np.zeros([6,4]) # empty matrix to store amplitude values

colour_correlation_values = np.zeros([6,3]) # empty matrix to store amplitude values


# Load data
print('  ')
print('Loading EOG data ...')
print('  ')



# high pass filter the data to remove slow drifts

sample_rate = 5000

high_pass_filter = signal.butter(2, 0.1, 'hp', fs=sample_rate, output='sos')
VEOG_data = signal.sosfilt(high_pass_filter, VEOG_data)
HEOG_data = signal.sosfilt(high_pass_filter, HEOG_data)

        # low pass filter 
low_pass_filter = signal.butter(3, 5, 'lp', fs=sample_rate, output='sos')
VEOG_data = signal.sosfilt(low_pass_filter,VEOG_data)
HEOG_data = signal.sosfilt(low_pass_filter, HEOG_data)


print('Loading data ...')
print('  ')

start_time_electrode = time.process_time() # keep track of how long the script takes for each electrode 



#### filters


#     # high pass filter the data to remove slow drifts
high_pass_filter = signal.butter(2, 10, 'hp', fs=sample_rate, output='sos')
data = signal.sosfilt(high_pass_filter, data)

#         # low pass filter 
# low_pass_filter = signal.butter(1, 100, 'lp', fs=sample_rate, output='sos')
# data = signal.sosfilt(low_pass_filter, data)


plot_count = 1

for freq_count in range(0,6): # loop through each frequency
    
    flicker_period = flicker_periods[freq_count]
    
    print('  ')
    print('Flicker period: ' + str(flicker_period) + ' ms  (max. number of segments = ' + str(int(60 * round(1000/flicker_period))) + ')')
    
    for colour in ('red', 'green', 'blue', 'white'): 
        
        print(colour)
       
        # load the triggers for this colour
        all_triggers = np.load(file_name + '_all_' + colour + '_triggers.npy')
        
        # empty matrix to put segments of data into
        segment_matrix = np.zeros([3600,flicker_period*5]) 
    
    
        seg_count = 0

        for k in range(0,len(all_triggers)-1):
            
            trigger_time = all_triggers[k]
            
            next_trigger_time = all_triggers[k+1]

            difference = next_trigger_time - trigger_time 
            

            if np.abs(difference/5 - flicker_period) <= 0.1: # if time between triggers maches the flicker period
                
                
            
                segment = data[trigger_time-offset:trigger_time+(flicker_period*5)-offset]
                
                ### check for eye blinks 
                
                VEOG_segment = VEOG_data[trigger_time-500:trigger_time+500]
                HEOG_segment = HEOG_data[trigger_time-500:trigger_time+500]
                
                
                ### plot all segments of data to look for noisy segments. Comment out plots when running the full script
               # segment = segment - segment.mean() # baseline correct, only necessary when plotting
                
                if (np.ptp(segment) < 0.00005) and (np.ptp(VEOG_segment) < 0.0001) and(np.ptp(HEOG_segment) < 0.0001):           
                    #plt.plot(segment,'c')
                    seg_count += 1
                    segment_matrix[seg_count,:] = segment
               # else:
                  #   plt.plot(segment,'r')
               
   
        print('Good segments = ' + str(seg_count)) 
           
        
        SSVEP = segment_matrix[0:seg_count,:].mean(axis=0) # average the segments to make the SSVEP
     
        SSVEP = SSVEP - SSVEP.mean() # baseline correct
    
        SSVEP = SSVEP * 1000000 # convert to micro volts
    
    
        ### linear interpolation
        SSVEP[0:12] = np.linspace(SSVEP[0], SSVEP[12], num=12)
        
        trigger_2_time = trigger_2_times[freq_count] + offset
        SSVEP[trigger_2_time:trigger_2_time+12] = np.linspace(SSVEP[trigger_2_time], SSVEP[trigger_2_time+12], num=12)
        
        
        # save the SSVEP
        if colour == 'red':
            SSVEP_red = np.copy(SSVEP) 
        elif colour == 'green':
            SSVEP_green = np.copy(SSVEP) 
        elif colour == 'blue':
            SSVEP_blue = np.copy(SSVEP)     
        elif colour == 'white':
            SSVEP_white = np.copy(SSVEP)  
  
    
    ## save amplitude values

    colour_amplitude_values[freq_count,0] = np.ptp(SSVEP_red) 
    colour_amplitude_values[freq_count,1] = np.ptp(SSVEP_green) 
    colour_amplitude_values[freq_count,2] = np.ptp(SSVEP_blue) 
    colour_amplitude_values[freq_count,3] = np.ptp(SSVEP_white)
  
    # ## save correlation values 
   
    red_white_corr_matrix = np.corrcoef(SSVEP_red, SSVEP_white)
    colour_correlation_values[freq_count,0]  = red_white_corr_matrix[0,1]
    
    green_white_corr_matrix = np.corrcoef(SSVEP_green, SSVEP_white)
    colour_correlation_values[freq_count,1]  = green_white_corr_matrix[0,1]  
    
    blue_white_corr_matrix = np.corrcoef(SSVEP_blue, SSVEP_white)
    colour_correlation_values[freq_count,2]  = blue_white_corr_matrix[0,1]  
    
  
## plots

    time_vector = np.arange(0,flicker_period, 1000/sample_rate)

    plt.subplot(2,3,plot_count)
    
    plt.plot(time_vector,SSVEP_red, 'r', label = 'Red')
    plt.plot(time_vector,SSVEP_green, 'g', label = 'Green')
    plt.plot(time_vector,SSVEP_blue, 'b', label = 'Blue')
    plt.plot(time_vector,SSVEP_white, 'k', label = 'White')
    

    if plot_count > 3: # only label the x axis on the bottom row
        plt.xlabel('Time (ms)')
    
    plt.title(str(round(1000/flicker_period)) + ' Hz')
    
    # plt.ylim(-1.6, 1.6)
    plt.ylim(-2.7, 3.5)
    
    plot_count += 1
    
plt.legend()

plt.suptitle('Subject: ' + str(subject) + '  Electrode: ' + electrode_name)

    

    
 
total_elapsed_time = time.process_time() - start_time

print('Total elapsed time = ' + str(np.round(total_elapsed_time)) + ' seconds')  
    
    
elapsed_time_electrode = time.process_time() - start_time_electrode

print('Elapsed time for one electrode = ' + str(np.round(elapsed_time_electrode)) + ' seconds')  
       
