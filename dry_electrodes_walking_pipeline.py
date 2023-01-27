#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 16:17:43 2022

@author: James Dowsett
"""

#########  Analysis of Dry electrode walking experiment  ####################



from flicker_analysis_package import functions

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats
from random import choice
import statistics
from scipy import signal

from timeit import default_timer as timer
from datetime import timedelta


path = '/home/james/Active_projects/mentalab_dry_electrodes/mentalab_test/subject_data/'

electrode_names = ('EOG', 'C5 sigi', 'C6 sigi', 'Dry ref FCz', 'P3 Dry', 'P4 dry', 'P5 gel', 'P6 gel')

condition_names = ('sigi_stand', 'sigi_walk', 'flicker_stand', 'flicker_walk', 'blackout')


sample_rate = 1000



period = 111


subjects_to_use = (1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 18, 20, 21, 22)

num_subjects = len(subjects_to_use)


num_electrodes = 7  # dry reference, 2 gel, 2 dry, 2 dry with re-reference to dry reference


#### matricies to store results  #################
walking_frequencies = np.zeros([num_subjects,2]) # get frequency from the X (vertical) axis, for the two walking conditions

walking_acc_amplitudes = np.zeros([num_subjects,2]) # get average peak to peak amplitude from the X (vertical) axis, for the two walking conditions

all_SSVEPs = np.zeros([num_subjects, num_electrodes,5, period]) # subject, electrode, condition

self_correlation_scores = np.zeros([num_subjects, num_electrodes,5]) # 50/50 split self correlation- average scores

SNR_scores = np.zeros([num_subjects, num_electrodes,5]) # Signal to noise ratio with the random shuffle peak-to-peak amplitude method


###################



subject_count = 0

for subject in subjects_to_use:
    print('Subject ' + str(subject))


    # get Accelerometer data
    
    # accelerometer_x_file_name = path + 'subject_' + str(subject) + '/subject_' + str(subject) + '_acc_chan_1_data.npy'    
    # accelerometer_x_data = np.load(accelerometer_x_file_name)
    
    # accelerometer_y_file_name = path + 'subject_' + str(subject) + '/subject_' + str(subject) + '_acc_chan_2_data.npy'  
    # accelerometer_y_data = np.load(accelerometer_y_file_name)
    
    # accelerometer_z_file_name = path + 'subject_' + str(subject) + '/subject_' + str(subject) + '_acc_chan_3_data.npy'  
    # accelerometer_z_data = np.load(accelerometer_z_file_name)
    
    # plot_scale = 1 #1000
    # plt.plot(accelerometer_x_data*plot_scale)
    # plt.plot(accelerometer_y_data*plot_scale)
    # plt.plot(accelerometer_z_data*plot_scale)
    
    # get gyroscope data
    
    # gyroscope_x_file_name = path + 'subject_' + str(subject) + '/subject_' + str(subject) + '_acc_chan_4_data.npy'     
    # gyroscope_x_data = np.load(gyroscope_x_file_name)
    
    # gyroscope_y_file_name = path + 'subject_' + str(subject) + '/subject_' + str(subject) + '_acc_chan_6_data.npy'     
    # gyroscope_y_data = np.load(gyroscope_y_file_name)
    
    # gyroscope_z_file_name = path + 'subject_' + str(subject) + '/subject_' + str(subject) + '_acc_chan_5_data.npy'  
    # gyroscope_z_data = np.load(gyroscope_z_file_name)    
    

    # plt.plot(gyroscope_x_data)
    # plt.plot(gyroscope_y_data)
    # plt.plot(gyroscope_z_data)
    

    # FFTs of accelerometer data
    
  #  plt.figure(subject)
    
    
    # length = 10 # length of FFT segment for walking frequency, 10 seconds = 0.1 Hz resolution
    
    # for condition in range(0,5):  ## (1,3):  #

    #         ## load triggers for condition
            
    #         condition_name = condition_names[condition]
            
    #      #   print(electrode_names[electrode])
    #         print(condition_name)
            
    #         trigger_file_name = path + 'subject_' + str(subject) + '/Subject_' + str(subject) + '_all_triggers_' + condition_name + '.npy'
            
    #         triggers = np.load(trigger_file_name)
          
    #         print(str(len(triggers)) + ' triggers')  
    #         print(' ')
    
            
    #         acc_fft_x = functions.induced_fft(accelerometer_x_data, triggers, length, sample_rate)
    #         # acc_fft_y = functions.induced_fft(accelerometer_y_data, triggers, length, sample_rate)
    #         # acc_fft_z = functions.induced_fft(accelerometer_z_data, triggers, length, sample_rate)
    
    #       #  time_axis = np.arange(0,1/length*len(acc_fft_x),1/length) # time_vector to plot FFT  
    
    #         # plt.subplot(1,3,1)
    #         # plt.title('X')
    #         # plt.plot(time_axis,acc_fft_x, label = condition_name)
    #         # plt.xlim([0, 10])
    #         # plt.subplot(1,3,2)
    #         # plt.title('Y')
    #         # plt.plot(time_axis,acc_fft_y, label = condition_name)
    #         # plt.xlim([0, 10])
    #         # plt.subplot(1,3,3)
    #         # plt.title('Z')
    #         # plt.plot(time_axis,acc_fft_z, label = condition_name)
    #         # plt.xlim([0, 10])
    
    #         # save the walking frequency for the two walking conditions
    #         if condition == 1:
    #             walking_frequencies[subject_count, 0] = (np.argmax(acc_fft_x[5:25]) + 5)/10
    #         elif condition == 3:
    #              walking_frequencies[subject_count, 1] = (np.argmax(acc_fft_x[5:25]) + 5)/10 
    #           #   plt.suptitle('Subject ' + str(subject) + '  X freq. = ' + str((np.argmax(acc_fft_x[5:25]) + 5)/10 ))
    
    
    #         ## get average peak to peak amplitude from one second segments of X (vertical) chan
    #         t = triggers[0]
    #         ptp_amplitudes = []
    #         while t < triggers[-1]:
    #             segment = accelerometer_x_data[t:t+1000]
    #             amplitude = np.ptp(segment)
    #             ptp_amplitudes.append(amplitude)
    #             t = t + 1000
            
    #         average_ptp_amplitude = statistics.mean(ptp_amplitudes)
 
    #         if condition == 1:
    #             walking_acc_amplitudes[subject_count, 0] = average_ptp_amplitude
    #         elif condition == 3:
    #             walking_acc_amplitudes[subject_count, 1] = average_ptp_amplitude

    
    
   # plt.legend()
    
    
    
    ######## EEG data ##########
    
    # plt.figure(subject)
    # plt.suptitle('Subject ' + str(subject))
    
    plot_count = 0
    
    electrode_count = 0
    
    for electrode in range(3,10): ##(4,5):  ### 
    
       
        
        # plot all electrodes, don't include dry reference
        # if electrode > 3:
        #     plot_count += 1
        #     plt.subplot(2,2,plot_count)
        #     plt.title(electrode_names[electrode])
        
    # load data, all conditions should be one data file
    
        if electrode < 8:
            
             print(electrode_names[electrode])
             print(' ')
             
             file_name = path + 'subject_' + str(subject) + '/subject_' + str(subject) + '_chan_' + str(electrode) + '_data.npy'
        
             data = np.load(file_name)
    
        elif electrode == 8:
            
            file_name = path + 'subject_' + str(subject) + '/subject_' + str(subject) + '_chan_4_data.npy'
            data = np.load(file_name)
            
            dry_ref_file_name = path + 'subject_' + str(subject) + '/subject_' + str(subject) + '_chan_3_data.npy'
            dry_ref = np.load(dry_ref_file_name)
            
            data = data - dry_ref # re-reference to dry reference
            
        elif electrode == 9:
            
            file_name = path + 'subject_' + str(subject) + '/subject_' + str(subject) + '_chan_5_data.npy'
            data = np.load(file_name)
            
            dry_ref_file_name = path + 'subject_' + str(subject) + '/subject_' + str(subject) + '_chan_3_data.npy'
            dry_ref = np.load(dry_ref_file_name)
            
            data = data - dry_ref # re-reference to dry reference
            
        
        #### filters  ####
    
        ## notch filter, first 50 Hz
        b, a = signal.iirnotch(50, 30, sample_rate)
        data = signal.filtfilt(b, a, data)
        
        # # then 100 Hz
        b, a = signal.iirnotch(100, 30, sample_rate)
        data = signal.filtfilt(b, a, data)
    
    
        # then 150 Hz
        b, a = signal.iirnotch(150, 30, sample_rate)
        data = signal.filtfilt(b, a, data)
      
        
      
    
        ## plot raw data
        # plt.figure(1)
        # data = data-data.mean()
        # plt.plot(data, label = electrode_names[electrode])
        # plt.legend()

        
        condition_count = 0
        
        for condition in range(0,5):  ## (1,3):  #
            
          
        
            ## load triggers for condition
            
            condition_name = condition_names[condition]
            
         #   print(electrode_names[electrode])
            print(condition_name)
            
            trigger_file_name = path + 'subject_' + str(subject) + '/Subject_' + str(subject) + '_all_triggers_' + condition_name + '.npy'
            
            triggers = np.load(trigger_file_name)
          
            print(str(len(triggers)) + ' triggers')  
            print(' ')
          
            
            ## linear interpolation
            
            if condition > 2: # only need linear interpolation on flicker and vlackout conditions, there is no trigger artefact for SIGI
                
                time_1 = -1
                time_2 = 56
                trig_length = 10
                
                data = functions.linear_interpolation(data, triggers, time_1, time_2, trig_length)
                
            
            ## make a time series of the triggers to visualize
            # plt.figure(1)
            # trigger_time_series = np.zeros([len(data)],)
             
            # for trigger in triggers:
            #     trigger_time_series[trigger] = 2000
                
            # data = data - data.mean()
            # plt.plot(data)
            # plt.plot(trigger_time_series)
            
                
                
            ########### make SSVEP  ##############
            
            SSVEP = functions.make_SSVEPs(data, triggers, period) # 
        
            all_SSVEPs[subject_count, electrode_count, condition_count,:] = SSVEP
                   
            # ### plot SSVEP 
            # if electrode > 3: # plot all electrodes, don't include dry reference
            #     plt.plot(SSVEP, label = condition_names[condition])
            
            
            ## get self correlation score for SSVEP, average over 100 loops
            num_loops = 100
            scores = np.zeros([num_loops,])
            for loop in range(0,num_loops):
                self_correlation_score = functions.compare_SSVEPs_split(data, triggers, period)
                scores[loop] = self_correlation_score
                
            average_correlation = scores.mean()
            print(average_correlation)
            print(' ')

            self_correlation_scores[subject_count, electrode_count, condition_count] = average_correlation
            
      
            ## get Signal to Noise ratio with random shuffle function

            scores = np.zeros([num_loops,])
            for loop in range(0,num_loops):
                scores[loop] = functions.SNR_random(data, triggers, period)
      

            SNR_scores[subject_count, electrode_count, condition_count] = scores.mean()
      
            
      
            condition_count += 1  
      
        electrode_count +=1       
        
              
   # plt.legend()
    
    #np.save(path + 'all_clean_segments',all_clean_segments)
    
    subject_count += 1 


## save


np.save(path + 'all_SSVEPs', all_SSVEPs)

np.save(path + 'self_correlation_scores', self_correlation_scores)

np.save(path + 'SNR_scores', SNR_scores)



# np.save(path + 'walking_frequencies', walking_frequencies)

# np.save(path + 'walking_acc_amplitudes', walking_acc_amplitudes)


##  load saved data

all_SSVEPs = np.load(path + 'all_SSVEPs.npy')  # subject, electrode, condition, data

self_correlation_scores = np.load(path + 'self_correlation_scores.npy')  # subject, electrode, condition

SNR_scores = np.load(path + 'SNR_scores.npy')


## plot self correlations

electrode_names = ('Left Dry', 'Right Dry', 'Left Gel', 'Right Gel')

condition_colours = ('b', 'c', 'r', 'm', 'k')

small_dot_size = 2

for noise_type in range(0,2):
    
    plt.figure()
    
    if noise_type == 0:
        data_to_plot = np.copy(self_correlation_scores)
        plt.title('Self correlation')
    elif noise_type == 1:
        data_to_plot = np.copy(SNR_scores)
        plt.title('Average SNR')        
    
    
    for electrode in range(1,5):
        
    
        for condition in range(0,5):
            
            mean_data = data_to_plot[:,electrode, condition].mean(axis=0)
            std_data = np.std(data_to_plot[:,electrode, condition])
    
            std_error_data = std_data / math.sqrt(len(subjects_to_use))
            
            plt.errorbar(electrode+(condition/10), mean_data,yerr = std_error_data, solid_capstyle='projecting', capsize=5,  fmt='o', color= condition_colours[condition], ecolor= condition_colours[condition])  
    
            plt.scatter(np.zeros([num_subjects,]) + electrode+(condition/10),data_to_plot[:,electrode, condition], color= condition_colours[condition], s=small_dot_size)
    
    # final plots
    
    x = np.arange(1,5) + 0.2
    plt.xticks(x, electrode_names)    
    plt.plot(1,0, 'b', label = 'Standing Signal Generator')
    plt.plot(1,0, 'c', label = 'Walking Signal Generator')
    plt.plot(1,0, 'r', label = 'Standing Flicker')
    plt.plot(1,0, 'm', label = 'Walking Flicker')
    plt.plot(1,0, 'k', label = 'Blackout')
    
    plt.legend()
    
    if noise_type == 0:
        plt.ylabel('Average Self correlation')
    elif noise_type == 1:
        plt.ylabel('SNR')


## plot  SSVEPs

for electrode in range(1,5):
    
    plt.figure()
    plt.suptitle(electrode_names[electrode-1])
    
    for subject in range(0,18):
        
        plt.subplot(4,5,subject+1)
        plt.title(subject+1)

        for condition in range(0,5):
            SSVEP = all_SSVEPs[subject,electrode,condition,:] 
            plt.plot(SSVEP)
        
        # for electrode in range(1,5):
            
        #     blackout_SSVEP = all_SSVEPs[subject,electrode,4,:] 
        
        #     plt.plot(blackout_SSVEP)    
    


## plot SSVEPs comparing gel and dry standing

for condition in ('SIGI', 'Flicker'):
    
    if condition == 'SIGI':
        condition_num = 0
    elif condition =='Flicker':
        condition_num = 2
        

    for hemisphere in ('Left', 'Right'):
    
        plt.figure()
        plt.suptitle(condition + ' ' + hemisphere)
    
        for subject in range(0,18):
            
            plt.subplot(4,5,subject+1)
            
        
        
            if hemisphere == 'Left':
                
                gel_SSVEP = all_SSVEPs[subject,3,condition_num,:] 
                dry_SSVEP = all_SSVEPs[subject,1,condition_num,:] 
                
            elif hemisphere =='Right':
                
                gel_SSVEP = all_SSVEPs[subject,4,condition_num,:] 
                dry_SSVEP = all_SSVEPs[subject,2,condition_num,:] 
                
            plt.plot(gel_SSVEP, label = 'Gel')
            plt.plot(dry_SSVEP,label = 'Dry')        
    
    
            correlation = np.corrcoef(gel_SSVEP, dry_SSVEP)[0,1]           
    
            plt.title(str(subject+1) + ' ' + str(np.round(correlation,3)))
            
            
    plt.legend()        
    
    
    


