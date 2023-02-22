#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 13:44:34 2023

@author: James Dowsett
"""

#### Coma patients flicker analysis pipeline ####################



from flicker_analysis_package import functions

import numpy as np
import matplotlib.pyplot as plt
import mne



### information about the experiment:

path = '/media/james/USB DISK/coma_flicker_data/flicker_coma_data_02_08_22/' # put path here


frequency_names = ('30 Hz', '40 Hz')

sample_rate = 5000


period_30Hz = 166 # period of flicker in data points
period_40Hz = 125 # 


# load file names and folders
file_names = np.load(path + 'file_names_used.npy')


num_trials = len(file_names)


start_times = np.load(path + 'start_times.npy')

plt.figure(1)
plt.figure(2)

plot_count = 0

for trial in range(0,len(file_names)):
    
    print('\n Trial: ' + str(trial) + '\n')
    
    plot_count += 1

    #plt.figure()

    file_name = file_names[trial]
    
    
    ## load triggers
    
    triggers = np.load(path + 'file_' + str(trial) + '_triggers.npy')
    
    
    
    ## load data
    
    raw = mne.io.read_raw_brainvision(path + file_name, preload=True)
    
    chan_names = raw.ch_names
    
    photo_data = np.array(raw[34,:], dtype=object) 
    photo_data = photo_data[0,]
    photo_data = photo_data.flatten()
      
    VEOG_data = np.array(raw[33,:], dtype=object) 
    VEOG_data = VEOG_data[0,]
    VEOG_data =VEOG_data.flatten()
      
    trigger_data = photo_data - VEOG_data
    
    #plt.plot(trigger_data)
    
    ## make trigger time series
    
    trigger_time_series = np.zeros([len(trigger_data),])
    for trigger in triggers:
        trigger_time_series[trigger] = 0.001
        
    #plt.plot(trigger_time_series)
    
    
    
    if '30Hz' in file_name:
        period = period_30Hz
        #plt.subplot(1,2,1)
        plt.figure(1)
        plt.subplot(6,6,plot_count)
        plt.title('30 Hz')
        plt.figure(2)
        plt.subplot(6,6,plot_count)
        plt.title('30 Hz')
       # plt.title('Trial ' + str(trial) + '  30Hz')
    elif '40Hz' in file_name:
        period = period_40Hz
       # plt.subplot(1,2,2)
        plt.figure(1)
        plt.subplot(6,6,plot_count)
        plt.title('40 Hz')
        plt.figure(2)
        plt.subplot(6,6,plot_count)
        plt.title('40 Hz')
        #plt.title('Trial ' + str(trial) + '  ' + file_name + '  40Hz')
    
    
    for electrode in (11, 16, 25):
    
        
        data = np.array(raw[electrode,:], dtype=object) 
        
        data = data[0,]
        
        data = data.flatten()
        
        
        SSVEP = functions.make_SSVEPs(data, triggers, period) 
        
        plt.figure(1)
        plt.subplot(6,6,plot_count)
        
        plt.plot(SSVEP, label = chan_names[electrode])
        
        
        length = 100
        induced_fft = functions.induced_fft(data, triggers, length, sample_rate) # length = length of segment to use in seconds (1/length = the frequency resolution), sample rate in Hz
            
        time_axis = np.arange(0,sample_rate,1/length)    
    
        plt.figure(2)
        plt.subplot(6,6,plot_count)
        
        plt.plot(time_axis,induced_fft)
        plt.xlim([0, 10])

  #  plt.legend()
  #  plt.ylim([-0.0000015, 0.0000015])
    
    
plt.legend()    