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


path = '/home/james/Active_projects/mentalab_dry_electrodes/mentalab_test/subject_data/'

electrode_names = ('EOG', 'C5 sigi', 'C6 sigi', 'Dry ref FCz', 'P3 Dry', 'P4 dry', 'P5 gel', 'P6 gel')

condition_names = ('sigi_stand', 'sigi_walk', 'flicker_stand', 'flicker_walk', 'blackout')


sample_rate = 1000

num_subjects = 10

period = 111




#for subject in (0,1,2,4,5,6,7,8,9):
    
    
subject = 1

print('Subject ' + str(subject))

plt.figure()
plt.suptitle('Subject ' + str(subject))

plot_count = 0

for electrode in (4,5,6,7):
    
    plt.figure(2)
    plot_count += 1
    plt.subplot(2,2,plot_count)
    plt.title(electrode_names[electrode])

# load data, all conditions should be one data file


    file_name = path + 'subject_' + str(subject+1) + '/subject_' + str(subject+1) + '_chan_' + str(electrode) + '_data.npy'
    
    data = np.load(file_name)

    plt.figure(1)
    data = data-data.mean()
    plt.plot(data, label = electrode_names[electrode])
    # plt.legend()
    
    
    # get Accelerometer data
    
    # accelerometer_x_file_name = path + 'subject_' + str(subject+1) + '/subject_' + str(subject+1) + '_acc_chan_1_data.npy'    
    # accelerometer_x_data = np.load(accelerometer_x_file_name)
        
    # accelerometer_y_file_name = path + 'subject_' + str(subject+1) + '/subject_' + str(subject+1) + '_acc_chan_2_data.npy'  
    # accelerometer_y_data = np.load(accelerometer_y_file_name)
        
    # accelerometer_z_file_name = path + 'subject_' + str(subject+1) + '/subject_' + str(subject+1) + '_acc_chan_3_data.npy'  
    # accelerometer_z_data = np.load(accelerometer_z_file_name)
      
    # # plot_scale = 1000
    # # plt.plot(accelerometer_x_data*plot_scale)
    # # plt.plot(accelerometer_y_data*plot_scale)
    # # plt.plot(accelerometer_z_data*plot_scale)
    
    # # get gyroscope data

    # gyroscope_x_file_name = path + 'subject_' + str(subject+1) + '/subject_' + str(subject+1) + '_acc_chan_4_data.npy'     
    # gyroscope_x_data = np.load(gyroscope_x_file_name)
      
    # gyroscope_y_file_name = path + 'subject_' + str(subject+1) + '/subject_' + str(subject+1) + '_acc_chan_6_data.npy'     
    # gyroscope_y_data = np.load(gyroscope_y_file_name)
          
    # gyroscope_z_file_name = path + 'subject_' + str(subject+1) + '/subject_' + str(subject+1) + '_acc_chan_5_data.npy'  
    # gyroscope_z_data = np.load(gyroscope_z_file_name)    
        

 
  #  plt.plot(gyroscope_x_data)
    # plt.plot(gyroscope_y_data)
    # plt.plot(gyroscope_z_data)
    
    
    
    for condition in range(0,5):
        
        ## load triggers for condition
        
        condition_name = condition_names[condition]
        
        trigger_file_name = path + 'subject_' + str(subject+1) + '/Subject_' + str(subject+1) + '_all_triggers_' + condition_name + '.npy'
        
        triggers = np.load(trigger_file_name)
      
        
        # plt.figure(1)
        # trigger_time_series = np.zeros([len(data)],)
         
        # for trigger in triggers:
        #     trigger_time_series[trigger] = 2000

        # plt.plot(trigger_time_series)
            
            
        ## make SSVEP the normal way
        
        SSVEP = functions.make_SSVEPs(data, triggers, period) # 
    
        plt.figure(2)
        plt.plot(SSVEP, label = condition_names[condition])
    
    
        # make SSVEP with motion artefact removal
        
        # segment_matrix = np.zeros([len(triggers),period])
        
        # first_trigger = triggers[0]
        # last_trigger = triggers[-1]
        # length_trial = last_trigger - first_trigger 
        
        # for trigger in triggers:
            
        #     # get the segment to be cleaned
        #     segment = data[trigger:trigger+period]
            
        #     segment = segment - segment.mean()
            
        #     segment_range = np.ptp(segment)
            
        #     # check the entire trial for segments which correlate well and are a similar amplitude
        #     corr_values = np.zeros([length_trial,])
        #     ptp_values = np.zeros([length_trial,])
        #     difference_amplitudes = np.zeros([length_trial,])
            
        #     count = 0
        #     k = first_trigger
        #     while k < last_trigger:
                
        #         temp_segment = data[k:k+period]
                
        #         corr_values[count] = np.corrcoef(segment,temp_segment)[0,1]
        #         ptp_values[count] = np.ptp(temp_segment)
        #         difference_amplitudes[count] = np.ptp(segment-temp_segment)
                
        #         k += 1
        #         count += 1
            
            
        #     plt.figure()
            
        #     temp_segment_matrix = np.zeros([length_trial,period])
        #     segs_to_use_count = 0
        #     for k in range(0,length_trial):
        #         if corr_values[k] > 0.98 and difference_amplitudes[k] < 2000:
        #             temp_segment = data[first_trigger+k:first_trigger+k+period]
        #             temp_segment = temp_segment - temp_segment.mean()
        #             temp_segment_matrix[segs_to_use_count,:] = temp_segment
        #             segs_to_use_count += 1
        #             plt.plot(temp_segment,'r')
               
             
        #     average_temp_segments = temp_segment_matrix[0:segs_to_use_count,:].mean(axis = 0)    
        #     plt.plot(average_temp_segments,'g')
        #     plt.plot(segment,'b')
        #     plt.title(segs_to_use_count)
            
        #     clean_segment = segment - average_temp_segments
        #     plt.plot(clean_segment,'g')
    
        
plt.legend()

