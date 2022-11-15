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


from timeit import default_timer as timer
from datetime import timedelta


path = '/home/james/Active_projects/mentalab_dry_electrodes/mentalab_test/subject_data/'

electrode_names = ('EOG', 'C5 sigi', 'C6 sigi', 'Dry ref FCz', 'P3 Dry', 'P4 dry', 'P5 gel', 'P6 gel')

condition_names = ('sigi_stand', 'sigi_walk', 'flicker_stand', 'flicker_walk', 'blackout')


sample_rate = 1000



period = 111


subjects_to_use = (1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 18, 20, 21, 22)

num_subjects = len(subjects_to_use)

for subject in subjects_to_use:
    print('Subject ' + str(subject))


    # get Accelerometer data
    
    accelerometer_x_file_name = path + 'subject_' + str(subject) + '/subject_' + str(subject) + '_acc_chan_1_data.npy'    
    accelerometer_x_data = np.load(accelerometer_x_file_name)
    
    accelerometer_y_file_name = path + 'subject_' + str(subject) + '/subject_' + str(subject) + '_acc_chan_2_data.npy'  
    accelerometer_y_data = np.load(accelerometer_y_file_name)
    
    accelerometer_z_file_name = path + 'subject_' + str(subject) + '/subject_' + str(subject) + '_acc_chan_3_data.npy'  
    accelerometer_z_data = np.load(accelerometer_z_file_name)
    
    # plot_scale = 1 #1000
    # plt.plot(accelerometer_x_data*plot_scale)
    # plt.plot(accelerometer_y_data*plot_scale)
    # plt.plot(accelerometer_z_data*plot_scale)
    
    # get gyroscope data
    
    gyroscope_x_file_name = path + 'subject_' + str(subject) + '/subject_' + str(subject) + '_acc_chan_4_data.npy'     
    gyroscope_x_data = np.load(gyroscope_x_file_name)
    
    gyroscope_y_file_name = path + 'subject_' + str(subject) + '/subject_' + str(subject) + '_acc_chan_6_data.npy'     
    gyroscope_y_data = np.load(gyroscope_y_file_name)
    
    gyroscope_z_file_name = path + 'subject_' + str(subject) + '/subject_' + str(subject) + '_acc_chan_5_data.npy'  
    gyroscope_z_data = np.load(gyroscope_z_file_name)    
    

    # plt.plot(gyroscope_x_data)
    # plt.plot(gyroscope_y_data)
    # plt.plot(gyroscope_z_data)
    

    # FFTs of accelerometer data
    
    length = 10
    
    for condition in range(0,5):  ## (1,3):  #

            ## load triggers for condition
            
            condition_name = condition_names[condition]
            
         #   print(electrode_names[electrode])
            print(condition_name)
            
            trigger_file_name = path + 'subject_' + str(subject) + '/Subject_' + str(subject) + '_all_triggers_' + condition_name + '.npy'
            
            triggers = np.load(trigger_file_name)
          
            print(str(len(triggers)) + ' triggers')  
            print(' ')
    
            
            acc_fft_x = functions.induced_fft(accelerometer_x_data, triggers, length, sample_rate)
            acc_fft_y = functions.induced_fft(accelerometer_y_data, triggers, length, sample_rate)
            acc_fft_z = functions.induced_fft(accelerometer_z_data, triggers, length, sample_rate)
    
            time_axis = np.arange(0,1/length*len(acc_fft_x),1/length) # time_vector to plot FFT  
    
            plt.subplot(1,3,1)
            plt.plot(time_axis,acc_fft_x, label = condition_name)
            plt.xlim([0, 10])
            plt.subplot(1,3,2)
            plt.plot(time_axis,acc_fft_y, label = condition_name)
            plt.subplot(1,3,3)
            plt.plot(time_axis,acc_fft_z, label = condition_name)
    
    plt.legend()
    
    
    ## EEG data
    
    plt.figure()
    plt.suptitle('Subject ' + str(subject))
    
    plot_count = 0
    
    electrode_count = 0
    
    for electrode in range(4,8): ##(4,5):  ### 
    
        print(electrode_names[electrode])
        print(' ')
        
        plot_count += 1
        plt.subplot(2,2,plot_count)
        plt.title(electrode_names[electrode])
    
    # load data, all conditions should be one data file
    
    
        file_name = path + 'subject_' + str(subject) + '/subject_' + str(subject) + '_chan_' + str(electrode) + '_data.npy'
        
        data = np.load(file_name)
    
    
        ### plot raw data
        # plt.figure(1)
        # data = data-data.mean()
        # plt.plot(data, label = electrode_names[electrode])
        # plt.legend()

        
        walking_condition_count = 0
        
        for condition in range(0,5):  ## (1,3):  #
            
          
        
            ## load triggers for condition
            
            condition_name = condition_names[condition]
            
         #   print(electrode_names[electrode])
            print(condition_name)
            
            trigger_file_name = path + 'subject_' + str(subject) + '/Subject_' + str(subject) + '_all_triggers_' + condition_name + '.npy'
            
            triggers = np.load(trigger_file_name)
          
            print(str(len(triggers)) + ' triggers')  
            print(' ')
          
            ## make a time series of the triggers to visualize
            # plt.figure(1)
            # trigger_time_series = np.zeros([len(data)],)
             
            # for trigger in triggers:
            #     trigger_time_series[trigger] = 2000
                
            # data = data - data.mean()
            # plt.plot(data)
            # plt.plot(trigger_time_series)
            
                
                
            ########### make SSVEP the normal way  ##############
            
            SSVEP = functions.make_SSVEPs(data, triggers, period) # 
        
                   
            # ### plot SSVEP
            
            plt.plot(SSVEP, label = condition_names[condition])
            
      
        electrode_count +=1       
        
              
    plt.legend()
    
    #np.save(path + 'all_clean_segments',all_clean_segments)
    
    
