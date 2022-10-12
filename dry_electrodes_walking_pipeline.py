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

num_subjects = 10

period = 111

### empty matrix for the cleaned SSVEPs from the 2 dry electrodes from the walking conditions (SIGI and flicker)

all_clean_segments = np.zeros([num_subjects,2,2,100000,period]) # 10 subjects, 2 dry electrodes, 2 walking conditions, segment number, segment data

# all_clean_segments = np.load(path + 'all_clean_segments.npy')

# for subject in (0,1,2,4,5,6,7,8,9):
    
    
subject = 1
    
print('Subject ' + str(subject))

# plt.figure()
# plt.suptitle('Subject ' + str(subject))

plot_count = 0

electrode_count = 0

for electrode in (4,5):  ### range(4,8): ##
    
  
    # plot_count += 1
    # plt.subplot(2,2,plot_count)
    # plt.title(electrode_names[electrode])

# load data, all conditions should be one data file


    file_name = path + 'subject_' + str(subject+1) + '/subject_' + str(subject+1) + '_chan_' + str(electrode) + '_data.npy'
    
    data = np.load(file_name)


    ### plot raw data
    # plt.figure(1)
    # data = data-data.mean()
    # plt.plot(data, label = electrode_names[electrode])
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
    
    
    
    
    walking_condition_count = 0
    
    for condition in (1,3):  #range(0,5):  ## 
        
        ## load triggers for condition
        
        condition_name = condition_names[condition]
        
        print(condition_name)
        
        trigger_file_name = path + 'subject_' + str(subject+1) + '/Subject_' + str(subject+1) + '_all_triggers_' + condition_name + '.npy'
        
        triggers = np.load(trigger_file_name)
      
        ## make a time series of the triggers to visualize
        # plt.figure(1)
        # trigger_time_series = np.zeros([len(data)],)
         
        # for trigger in triggers:
        #     trigger_time_series[trigger] = 2000

        # plt.plot(trigger_time_series)
            
            
        ## make SSVEP the normal way
        
        SSVEP = functions.make_SSVEPs(data, triggers, period) # 
    
        
        
    
    
        #### make SSVEP from the pre-cleaned segments
        
        # if (condition == 1 or condition == 3) and (electrode == 4 or electrode == 5):
        
        #     if condition == 1: 
        #         walking_condition_count = 0  
        #     elif condition == 3:
        #         walking_condition_count = 1
            
        #     if electrode == 4:
        #         electrode_count = 0
        #     elif electrode == 5:
        #         electrode_count = 1
            
            
        #     trig_count = 0
            
        #     segment = all_clean_segments[subject,electrode_count,walking_condition_count,trig_count,:] 
            
        #     while np.ptp(segment) > 0:
                
        #         trig_count += 1
                
        #         segment = all_clean_segments[subject,electrode_count,walking_condition_count,trig_count,:] 
                
                
                    
        #     all_segments = all_clean_segments[subject,electrode_count,walking_condition_count,0:trig_count,:] 
    
        #     SSVEP = all_segments.mean(axis=0) # average to make SSVEP
            
        #     SSVEP = SSVEP - SSVEP.mean() # baseline correct
        
        
        # ### plot SSVEP
        
        # plt.plot(SSVEP, label = condition_names[condition])
        
        
        
    
        ################  motion artefact removal  #################
        
        
        first_trigger = triggers[0]
        last_trigger = triggers[-1]

        length_of_artefact_segment = period * 5 # length of segments of data from which to remove the motion artefact from

        # timer
        start_time = timer() # start a timer to keep see how long the artefact removal is taking
        time_stamps = np.zeros([len(triggers),])
        
        
        trig_count = 0
        

        while trig_count < (len(triggers)-10): # loop untill 10 triggers before the end, because flicker is 9 Hz and artefact removal uses one second segments
            
            
            print(' ')
            print('Subject ' + str(subject) + ' ' + electrode_names[electrode] + '  ' + condition_name + '   Trigger ' + str(trig_count) + ' of ' + str(len(triggers)))
         
              ##### display estimated time remaining ##########################################
               
            print('  ')
               
  
            elapsed_time_in_seconds = timer() - start_time
               
            print('Time elapsed in seconds = ' + str(np.round(elapsed_time_in_seconds,1)))
               
            
               
            if trig_count > 2:
               
                time_per_trigger = elapsed_time_in_seconds/trig_count
               
                time_remaining_in_seconds = (len(triggers) - trig_count) * time_per_trigger
               
                time_remaining = timedelta(seconds=time_remaining_in_seconds)
               
                print('Average time per trigger = ' + str(np.round(time_per_trigger,2)))
               
                print('Estimated time remaining = ' + str(time_remaining))
            #################################################################################
                   
            
            k = triggers[trig_count]

            # get artefact segment 
            artefact_segment = data[k:k+length_of_artefact_segment]
            
            artefact_segment_start_time = triggers[trig_count]
            
            if np.ptp(artefact_segment) < 200: # if motion artefact is not sufficently large, ignore and just use the original data
                
                cleaned_artefact_segment = artefact_segment
                
            else:
                
               cleaned_artefact_segment = np.zeros([length_of_artefact_segment,])
               
               for t in range(0,length_of_artefact_segment-10):
                   
                   value_min = min(artefact_segment[t:t+10])
                   value_max = max(artefact_segment[t:t+10])
                    
                
            ## put the 9 segments from the one second segment into the clean segment matrix
            trigger_time = artefact_segment_start_time
            while trigger_time <= (artefact_segment_start_time + length_of_artefact_segment - period):
                
                segment = cleaned_artefact_segment[trigger_time - artefact_segment_start_time:trigger_time - artefact_segment_start_time+period]
                
                all_clean_segments[subject,electrode_count,walking_condition_count,trig_count,:] = segment
                
                trig_count += 1
                
                trigger_time = triggers[trig_count] 
                
                
            
            
         
        walking_condition_count += 1   

        
    electrode_count += 1
          
    
    
    
          
#plt.legend()

#np.save(path + 'all_clean_segments',all_clean_segments)


