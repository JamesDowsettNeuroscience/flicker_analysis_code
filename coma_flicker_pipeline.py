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
import math

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




############   SSVEPs    ########################

# select the patients to include in analysis
patient_numbers = np.load(path + 'patient_numbers.npy')

patient_numbers = np.sort(patient_numbers)

patient_numbers_to_use = [21, 24, 25,  30, 34, 35, 62] ## OK SSVEPs with average reference

#patient_numbers_to_use = [21, 25, 35, 41, 62] ## OK SSVEPs in lapacian

linear_interpolation_time_1_30Hz = [97, 97, 97, 90, 95, 96, 95]
linear_interpolation_time_2_30Hz = [142, 142, 143, 143, 142, 143, 144]

linear_interpolation_time_1_40Hz = [-1, -1, -1, -1, -1, -19, -1]
linear_interpolation_time_2_40Hz = [81,81,78, 82, 80, 80, 80]


SSVEP_amplitudes = np.zeros([len(patient_numbers_to_use),2])

patient_count = 0

for patient_number in patient_numbers_to_use:
    
    plt.figure()
    plt.suptitle('Patient ' + str(patient_number) + '  ' + str(patient_count))   
    
      
    trials = []
    
    for trial_name in trial_names:
        if (str(patient_number) + '_') in trial_name:
            trials.append(trial_name)
    
    
    if patient_number == 34: # patient 34 trial 1 data is corrupted, do not use
        trials.remove('Patient_34_trial_1_30Hz')
        trials.remove('Patient_34_trial_1_40Hz')
        
    
    if patient_number == 30: # patient 30 both trial were OK, just use trial 0 for now
        trials.remove('Patient_30_trial_1_30Hz')
        trials.remove('Patient_30_trial_1_40Hz')   
        
    
    print('\n' + str(patient_number))
    print(trials)
    print(' ')
    
    for trial_name in trials:

        ## load triggers
        
        triggers = np.load(path + trial_name + '_triggers.npy')
 
        ## load data
        if laplacian == 0:
            all_data = np.load(path + trial_name + '_all_data.npy')

            average_reference = all_data.mean(0)
            #all_data = all_data - all_data.mean(0)
            
        elif laplacian == 1:
            all_data = np.load(path + trial_name + '_all_data_laplacian.npy')
        
     
        
        ## make trigger time series
        
        trigger_time_series = np.zeros([np.shape(all_data)[1],])
        for trigger in triggers:
            trigger_time_series[trigger] = 0.001
            
        #plt.plot(trigger_time_series)
        
        
        
        if '30Hz' in trial_name:
            period = period_30Hz
            plt.subplot(1,2,1)
            plt.title('30 Hz')
            trig_1_time = linear_interpolation_time_1_30Hz[patient_count]
            trig_2_time = linear_interpolation_time_2_30Hz[patient_count]
           
        elif '40Hz' in trial_name:
            period = period_40Hz
            plt.subplot(1,2,2)
            plt.title('40 Hz')
    
            trig_1_time = linear_interpolation_time_1_40Hz[patient_count]
            trig_2_time = linear_interpolation_time_2_40Hz[patient_count]
        
        
        
        electrode = 48  #electrodes_to_use[electrode_count]  ### 48 = POz
        #for electrode in range(0,64):  #(11, 16, 25):
        
        
        data = all_data[electrode,:]
        
        # linear interpolation
        trig_length = 15
        if patient_number == 35 and '40Hz' in trial_name: # patient 35 needs a longer interpolation
            trig_length = 30

        data = functions.linear_interpolation(data, triggers, trig_1_time, trig_2_time, trig_length)
  
        if laplacian == 0:
            average_reference = functions.linear_interpolation(average_reference, triggers, trig_1_time, trig_2_time, trig_length)
            average_reference_SSVEP = functions.make_SSVEPs(average_reference, triggers, period) 
           
            SSVEP = functions.make_SSVEPs(data, triggers, period) 
            SSVEP = SSVEP-average_reference_SSVEP # re-reference
            
        elif laplacian == 1:
            
            plt.plot(SSVEP, label = trial_name)
            SSVEP = functions.make_SSVEPs(data, triggers, period) 
         
        ## record SSVEP amplitudes
        if '30Hz' in trial_name:
            SSVEP_amplitudes[patient_count,0] = np.ptp(SSVEP)
        elif '40Hz' in trial_name:
            SSVEP_amplitudes[patient_count,1] = np.ptp(SSVEP)
        

        plt.plot(SSVEP, label = trial_name)


            #plt.subplot(8,8,electrode+1)
           # plt.ylim([-0.00025, 0.00025])
       # plt.plot(average_reference_SSVEP)
       # 
           # plt.title(chan_names[electrode] + ' '  + str(electrode))

    plt.subplot(1,2,1)
    plt.legend()    
    plt.subplot(1,2,2)
    plt.legend()    

    patient_count += 1





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

true_peaks = np.zeros([len(patient_numbers_to_use),3])

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
    
    
    ## check if the patient had more than one session
    more_than_one_session = 0
    for trial_name in trials:
        if 'trial_1' in trial_name and (patient_number != 41):
            more_than_one_session = 1
            
    ## loop through all trials for this patient
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
        
        true_peak = fft_spectrum[frequency_index]
        
        if 'trial_0' in trial_name:
        
            if '30Hz' in trial_name:
                peak_values[patient_count,0] = average_peak_range
                true_peaks[patient_count,0] = true_peak
            elif '40Hz' in trial_name:
                peak_values[patient_count,1] = average_peak_range
                true_peaks[patient_count,1] = true_peak
            elif 'baseline' in trial_name:
                peak_values[patient_count,2] = average_peak_range
                true_peaks[patient_count,2] = true_peak
                        
        if 'trial_1' in trial_name:
        
            if '30Hz' in trial_name:
                peak_values_trial_1[patient_count,0] = average_peak_range
            elif '40Hz' in trial_name:
                peak_values_trial_1[patient_count,1] = average_peak_range
            elif 'baseline' in trial_name:
                peak_values_trial_1[patient_count,2] = average_peak_range
                                
        
        
        # set condition names for plot legend
        if '30Hz' in trial_name:
            condition_name = '30 Hz Flicker'
        elif '40Hz' in trial_name:
            condition_name = '40 Hz Flicker'
        elif 'baseline' in trial_name:
           condition_name = 'Baseline'
                            
    
        
        ## plots
        if more_than_one_session == 1:
            if 'trial_0' in trial_name:
                plt.subplot(1,2,1)        
            if 'trial_1' in trial_name:
                plt.subplot(1,2,2)
            
       # plt.title(chan_names[electrode] )    
        plt.title(chan_names[electrode] + '  ' + str(dominant_frequency[0,0]) + ' Hz')    
        
        plt.plot(freq_vector,fft_spectrum, label = condition_name)

        max_peak = max(true_peaks[patient_count,:]) * 1.5

        plt.ylim([0, max_peak])
        plt.xlim([2, 20])
        
        plt.axvline(x = dominant_frequency, color = 'k', linestyle='--')
        # plt.axvline(x = dominant_frequency-1, color = 'k', linestyle='-.')
        # plt.axvline(x = dominant_frequency+1, color = 'k', linestyle='-.')   
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
            

    if more_than_one_session == 1:
        plt.subplot(1,2,1)
        plt.legend()
        plt.subplot(1,2,2)
        plt.legend()
    else:
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


## plot group data

plt.figure()

dot_size = 10

std_error_scores_30Hz = np.std(scores_as_percentage_baseline_30Hz) / math.sqrt(len(scores_as_percentage_baseline_30Hz))

plt.errorbar(0, scores_as_percentage_baseline_30Hz.mean(),yerr = std_error_scores_30Hz, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'b', ecolor='b')  

plt.scatter((np.zeros(len(scores_as_percentage_baseline_30Hz))), scores_as_percentage_baseline_30Hz, s=dot_size)


std_error_scores_40Hz = np.std(scores_as_percentage_baseline_40Hz) / math.sqrt(len(scores_as_percentage_baseline_40Hz))

plt.errorbar(1, scores_as_percentage_baseline_40Hz.mean(),yerr = std_error_scores_40Hz, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'b', ecolor='b')  

plt.scatter((np.ones(len(scores_as_percentage_baseline_40Hz))), scores_as_percentage_baseline_40Hz, s=dot_size)


plt.axhline(y = 100, color = 'k', linestyle='--')

plt.xticks((0,1), ('30 Hz', '40 Hz'))    
plt.xlim([-0.5,1.5])

plt.ylabel('Amplitude as percentage of Baseline')




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
        
        shuffled_values = np.zeros([len(peak_values),2])
        
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
######### Sleep mask pilot data as a control ##############

trigger_times = [-1, 59]


path = '/home/james/Active_projects/misc/Laura/sleep_mask_pilot_data/'

subject_numbers = (2,3,7,8)

POz_electrode_numbers = (62, 7, 7, 7)

plt.figure(1)
plt.figure(2)

for subject in range(0,len(subject_numbers)):
    
    folder_name = 'VP0' + str(subject_numbers[subject]) + '_P3/'

    for condition in range(0,2):
        
        if condition == 0:
            file_name = 'blackout'
            plot_name = 'Baseline'
        elif condition == 1:
            file_name = 'lux100'
            plot_name = '40 Hz Flicker'
    
    
        #    ### load raw data with MNE
        raw = mne.io.read_raw_brainvision(path + folder_name + file_name + '.vhdr', preload=True)
    
        ### Ch63=POz
        
        electrode_number = POz_electrode_numbers[subject]
        
        data = np.array(raw[electrode_number,:], dtype=object) 
        data = data[0,]
        data = data.flatten()
        
        
        length = 5
        
        fft_spectrum = functions.induced_fft(data, length, sample_rate)
        
        freq_vector = np.arange(0,len(fft_spectrum)/length,1/length)

        plt.figure(1)
        plt.subplot(2,2,subject+1)
        
        plt.plot(freq_vector, fft_spectrum, label = plot_name)
        
        plt.xlim([2, 45])
        plt.ylim([0, max(fft_spectrum[8*5:14*5]) * 1.2])
        
        
        #############  SSVEPs  ######################
        
        period = period_40Hz
        
        ### read triggers
        
        f = open(path + folder_name + file_name + '.vmrk') # open the .vmrk file, call it "f"
        
        # use readline() to read the first line 
        line = f.readline()
        
        # empty lists to record trigger times 
        triggers_list = []
       
        
        # the names of the triggers, these can be seen in the .vmrk file

        flicker_trigger_name = 'S  8'
        
        # use the read line to read further.
        # If the file is not empty keep reading one line
        # at a time, until the end
        while line:

            if flicker_trigger_name in line: # if the line contains a flicker trigger
                
                # get the trigger time from line
                start = line.find(flicker_trigger_name + ',') + len(flicker_trigger_name) + 1
                end = line.find(",1,")
                trigger_time = line[start:end]       
               
                # append the trigger time to the correct condition
                
                triggers_list.append(trigger_time)
                
            
            line = f.readline() # use realine() to read next line
            
        f.close() # close the file
        
        # convert to numpy arrays
        triggers = np.array(triggers_list, dtype=int)

        ### linear interpolation
        trig_length = 15
        data = functions.linear_interpolation(data, triggers, trigger_times[0], trigger_times[1], trig_length)

        # make SSVEP
        SSVEP = functions.make_SSVEPs(data, triggers, period) 
        
        plt.figure(2)    
        plt.subplot(2,2,subject+1)
        
        plt.plot(SSVEP, label = plot_name)
        
    plt.figure(1)    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()
    
    
    plt.figure(2)    
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.legend()
    
     
    
    