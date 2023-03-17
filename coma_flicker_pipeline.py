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

import pickle

### information about the experiment:


path = '/home/james/Active_projects/coma_flicker/extracted_files_and_triggers/'

frequency_names = ('30 Hz', '40 Hz')

sample_rate = 5000


period_30Hz = 166 # period of flicker in data points
period_40Hz = 125 # 


# load file names and folders
file_names = np.load(path + 'file_names_used.npy')


trial_names = pickle.load(open(path + "trial_names.pkl", "rb" ) )

chan_names = pickle.load(open(path + "chan_names.p", "rb" ) )

num_trials = len(trial_names)


start_times = np.load(path + 'start_times.npy')


laplacian = 0 ## choose to use laplacian montage or not: 0 = normal reference, 1 = laplacian re-reference


#plt.figure()

#############   SSVEPs    ########################

# plot_count = 0

# for trial_name in trial_names[30:35]:
    
#     print('\n '  + trial_name + '\n')
    
#     plot_count += 1

#     plt.figure()
#     plt.suptitle(trial_name)

    
#     ## load triggers
    
#     triggers = np.load(path + trial_name + '_triggers.npy')
    
    
#     ## load data
#     if laplacian == 0:
#         all_data = np.load(path + trial_name + '_all_data.npy')
        
#         average_reference = all_data.mean(0)
#         #all_data = all_data - all_data.mean(0)
        
#     elif laplacian == 1:
#         all_data = np.load(path + trial_name + '_all_data_laplacian.npy')
    
 
    
#     ## make trigger time series
    
#     trigger_time_series = np.zeros([np.shape(all_data)[1],])
#     for trigger in triggers:
#         trigger_time_series[trigger] = 0.001
        
#     #plt.plot(trigger_time_series)
    
    
    
#     if '30Hz' in trial_name:
#         period = period_30Hz
#         #plt.subplot(1,2,1)

#         # plt.subplot(6,6,plot_count)
#         # plt.title(trial_name)
#     elif '40Hz' in trial_name:
#         period = period_40Hz
#        # plt.subplot(1,2,2)

#         # plt.subplot(6,6,plot_count)
#         # plt.title(trial_name)
       
    
    
#     for electrode in range(0,64):  #(11, 16, 25):
    
        
#         data = all_data[electrode,:]
        
#         SSVEP = functions.make_SSVEPs(data, triggers, period) 
        
#         average_reference_SSVEP = functions.make_SSVEPs(average_reference, triggers, period) 

#         plt.subplot(8,8,electrode+1)
        
#         plt.plot(SSVEP)  #, label = chan_names[electrode])
#         plt.plot(average_reference_SSVEP)
#         plt.plot(SSVEP-average_reference_SSVEP)
#         plt.title(chan_names[electrode])


    
#plt.legend()    




#######################  FFT compared to resting   #####################################



# load file names and folders
baseline_trial_names = np.load(path + 'balseline_trial_names.npy')

## append the flicker and baseline file names into one array
all_trial_names = np.asarray(trial_names)
all_trial_names = np.append(all_trial_names,baseline_trial_names)

patient_numbers = np.load(path + 'patient_numbers.npy')

patient_numbers = np.sort(patient_numbers)


for patient_number in patient_numbers:
    
    plt.figure()
    plt.title('Patient ' + str(patient_number))
    
    trials = []
    
    for trial_name in all_trial_names:
        if (str(patient_number) + '_') in trial_name:
            trials.append(trial_name)
    
    print('\n' + str(patient_number))
    print(trials)
    print(' ')
    
    
    for trial_name in trials:
        
        all_data = np.load(path + trial_name + '_all_data.npy')
        
        electrode = 48  ### 48 = POz
        
        data = all_data[electrode,:]
        
        length = 5
        
        fft_spectrum = functions.induced_fft(data, length, sample_rate)
    
        time_vector = np.arange(0,len(fft_spectrum)/length,1/length)
    
        plt.plot(time_vector,fft_spectrum, label = trial_name)

    plt.ylim([0, fft_spectrum[2*length]])
    plt.xlim([2, 60])
    plt.legend()