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

# select the patients to include in analysis
patient_numbers = np.load(path + 'patient_numbers.npy')

patient_numbers = np.sort(patient_numbers)

patient_numbers_to_use = [21, 25, 29, 30, 31, 32, 34, 35, 38, 41, 62]

# load file names and folders
baseline_trial_names = np.load(path + 'balseline_trial_names.npy')

## append the flicker and baseline file names into one array
all_trial_names = np.asarray(trial_names)
all_trial_names = np.append(all_trial_names,baseline_trial_names)

## for ALL patients, manually identify the dominat oscillation
dominant_frequencies = np.zeros([len(patient_numbers),2])
dominant_frequencies[:,0] = patient_numbers
dominant_frequencies[:,1] = [ 6.4,  4.8,  5. ,  3. ,  4.2,  7.2,  4.4,  6.4,  7. ,  6.4,  6.6, 12.4,  1 ,  9.2]


peak_values = np.zeros([len(patient_numbers_to_use),3])
peak_values_trial_1 = np.zeros([len(patient_numbers_to_use),3])

patient_count = 0

for patient_number in patient_numbers_to_use:
    
    plt.figure()
    plt.suptitle('Patient ' + str(patient_number))
    
    trials = []
    
    for trial_name in all_trial_names:
        if (str(patient_number) + '_') in trial_name:
            trials.append(trial_name)
    
    print('\n' + str(patient_number))
    print(trials)
    print(' ')
    
    ## find the correct dominant frequency
    dominant_frequency = dominant_frequencies[np.where(dominant_frequencies[:,0] == patient_number),1]
    
    
    
    for trial_name in trials:
        
       # electrodes_to_use = [48, 1]
        
       # for electrode_count in range(0,2):

        electrode = 48  #electrodes_to_use[electrode_count]  ### 48 = POz
       
     #   plt.subplot(1,2,electrode_count+1)
        
             

        all_data = np.load(path + trial_name + '_all_data.npy')

        data = all_data[electrode,:]
        
        length = 5
        
        fft_spectrum = functions.induced_fft(data, length, sample_rate)
    
        freq_vector = np.arange(0,len(fft_spectrum)/length,1/length)
        
        
        ## store values
        
        frequency_index = int(dominant_frequency[0,] * length)
        
        range_index = length
        
        peak_range = fft_spectrum[frequency_index-range_index:frequency_index+range_index+1]
        
        average_peak_range = peak_range.mean()
        
        if 'trial_0' in trial_name:
        
            if '30Hz' in trial_name:
                peak_values[patient_count,0] = average_peak_range
            elif '40Hz' in trial_name:
                peak_values[patient_count,1] = average_peak_range
            elif 'baseline' in trial_name:
                peak_values[patient_count,2] = average_peak_range
                        
        if 'trial_1' in trial_name:
        
            if '30Hz' in trial_name:
                peak_values_trial_1[patient_count,0] = average_peak_range
            elif '40Hz' in trial_name:
                peak_values_trial_1[patient_count,1] = average_peak_range
            elif 'baseline' in trial_name:
                peak_values_trial_1[patient_count,2] = average_peak_range
                                
        
        
        
        ## plots
        
        
        plt.title(chan_names[electrode])    
        
        plt.plot(freq_vector,fft_spectrum, label = trial_name)

        plt.ylim([0, fft_spectrum[2*length]])
        plt.xlim([2, 60])
        
        plt.axvline(x = dominant_frequency, color = 'k', linestyle='--')
        plt.axvline(x = dominant_frequency-1, color = 'k', linestyle='-.')
        plt.axvline(x = dominant_frequency+1, color = 'k', linestyle='-.')   
            
    plt.legend()
    patient_count += 1
    
    
    
    
## sort peak values to use for analysis

peak_values[9,2] = peak_values_trial_1[9,2] # patient 41, baseline was pre recording and labeled as "trial_1"


peak_values[3,:] = peak_values_trial_1[3,:] # patient 30 no baseline in trial 0, use trial 1

trial_to_add = [peak_values_trial_1[4,:]] # patient 31 was good both trials, use both
peak_values = np.append(peak_values,trial_to_add, axis=0)

trial_to_add = [peak_values_trial_1[6,:]] # patient 34 was good both trials, use both
peak_values = np.append(peak_values,trial_to_add, axis=0)


scores_as_percentage_baseline_30Hz = (100/peak_values[:,2]) * peak_values[:,0] 
scores_as_percentage_baseline_40Hz = (100/peak_values[:,1]) * peak_values[:,0]     

average_30Hz = scores_as_percentage_baseline_30Hz.mean()
average_40Hz = scores_as_percentage_baseline_40Hz.mean()



## permutation tests

print('running permutation tests ...\n')

import random
import scipy.stats


num_loops = 100000

for freq_to_test in(0,1):# 0 = 30 Hz, 1 = 40 Hz
    
    if freq_to_test == 0:
        true_average = average_30Hz
    elif freq_to_test == 1: 
        true_average = average_40Hz
    
    average_shuffled_values = np.zeros([num_loops,])
    
    for loop in range(0,num_loops):
        
        shuffled_values = np.zeros([14,2])
        
        for k in range(0,len(peak_values)):
            
            if random.choice([0, 1]) == 0: # keep the same labels
                shuffled_values[k,0] = peak_values[k,freq_to_test]
                shuffled_values[k,1] = peak_values[k,2]
            else: # swap the labels
                shuffled_values[k,0] = peak_values[k,2]
                shuffled_values[k,1] = peak_values[k,freq_to_test]
                
                
        scores_as_percentage_baseline = (100/shuffled_values[:,1]) * shuffled_values[:,0] 
    
        average_shuffled_values[loop] = scores_as_percentage_baseline.mean()
    
    
   # plt.hist(average_shuffled_values)


    Z_score = (true_average - average_shuffled_values.mean()) / np.std(average_shuffled_values) # calculate Z score

    
    #find p-value for two-tailed test
    p_value = scipy.stats.norm.sf(abs(Z_score))*2
    
    if freq_to_test == 0:
        print('30 Hz Z score = ' + str(Z_score))
        
    elif freq_to_test == 1: 
        print('40 Hz Z score = ' + str(Z_score))

    print ('p = ' + str(p_value) + '\n')
