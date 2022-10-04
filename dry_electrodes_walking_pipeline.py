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

for subject in (0,1,2,4,5,6,7,8,9):
    
    
#subject = 1
    
    print('Subject ' + str(subject))
    
    # plt.figure()
    # plt.suptitle('Subject ' + str(subject))
    
    #plot_count = 0
    
    electrode_count = 0
    
    for electrode in (4,5):
        
        # plt.figure(2)
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
        
        for condition in (1,3): #range(0,5):
            
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
            
          #  SSVEP = functions.make_SSVEPs(data, triggers, period) # 
        
            # plt.figure(2)
            # plt.plot(SSVEP, label = condition_names[condition])
        
        
            ################  motion artefact removal  #################
            
          #  clean_segment_matrix = np.zeros([len(triggers),period])
            
            first_trigger = triggers[0]
            last_trigger = triggers[-1]
    
            
            
            # timer
            start_time = timer() # start a timer to keep see how long the artefact removal is taking
            time_stamps = np.zeros([len(triggers),])
            
            trig_count = 0
            for trigger in triggers:
                
                print(' ')
                print('Subject ' + str(subject) + ' ' + electrode_names[electrode] + '  ' + condition_name + '   Trigger ' + str(trig_count) + ' of ' + str(len(triggers)))
             
                 ##### display estimated time remaining ##########################################
                   
                print('  ')
                   
      
                elapsed_time_in_seconds = timer() - start_time
                   
                print('Time elapsed in seconds = ' + str(np.round(elapsed_time_in_seconds,1)))
                   
                time_stamps[trig_count] = timer() - start_time
                   
                if trig_count > 2:
                   
                   time_per_trigger = (np.diff(time_stamps[1:trig_count])).mean() 
                   
                   time_remaining_in_seconds = (len(triggers) - trig_count) * time_per_trigger
                   
                   time_remaining = timedelta(seconds=time_remaining_in_seconds)
                   
                   print('Average time per trigger = ' + str(np.round(time_per_trigger,2)))
                   
                   print('Estimated time remaining = ' + str(time_remaining))
                #################################################################################
                            
                
                
                
                # get the segment to be cleaned
                segment = data[trigger:trigger+period]
                
               # segment = segment - segment.mean()
                
                #segment_range = np.ptp(segment)
                
                # check the entire trial for segments which correlate well and are a similar amplitude
                
                # cutoff value above which to include the segment in the template
                corr_cutoff = 0.95
                # if the peak to peak amplitude of the difference between the segment to be cleaned and the template segment is greater than this value, ignore
                difference_cutoff = 200 
                
                max_num_template_segments = 50
                
                segment_locs = []
                
                # first look for segments before the trigger,
                if (trigger-30000) < first_trigger:  # if the first trigger was more than 30 seconds ago start from the first trigger
                    k = first_trigger 
                else:
                    k = trigger-30000 # if the first trigger was less than 30 seconds ago, start from 30 seconds ago
                
                while k < (trigger-200): # look up to 200 ms before the trigger time so we don't include the segment itself
                    
                    temp_segment = data[k:k+period]
                    if np.ptp(segment-temp_segment) < difference_cutoff:
                        if np.corrcoef(segment,temp_segment)[0,1] > corr_cutoff:
                           segment_locs.append(k)
                           
                    if len(segment_locs) > max_num_template_segments:
                        break
                    
                    k += 1
                    
                
                k = trigger + 200
                while k < (trigger+30000): # next look for segments after the trigger until 30 seconds after the trigger, skip the first200 ms after the trigger
                    
                    temp_segment = data[k:k+period]
                    if np.ptp(segment-temp_segment) < difference_cutoff:
                        if np.corrcoef(segment,temp_segment)[0,1] > corr_cutoff:
                           segment_locs.append(k)
                           
                    if len(segment_locs) > max_num_template_segments:
                        break
                    
                    k += 1               
                
                
              
                print('\n' + str(len(segment_locs)) + ' template segments')
                
                
                if len(segment_locs) > 10: # only subtract the template if there are at least 10 segments contributing to the template
                    
                    temp_segment_matrix = np.zeros([len(segment_locs),period])
                    seg_count = 0
                    for seg_time in segment_locs:
                       
                        temp_segment = data[seg_time:seg_time+period]
                       # temp_segment = temp_segment - temp_segment.mean()
                        temp_segment_matrix[seg_count,:] = temp_segment
                        seg_count += 1
                        #plt.plot(temp_segment,'r')
                   
                     
                    average_temp_segments = temp_segment_matrix[0:seg_count,:].mean(axis = 0)    
    
                    clean_segment = segment - average_temp_segments # clean the segment by subtracting the template
    
                   # clean_segment_matrix[trig_count,:] = clean_segment
                    all_clean_segments[subject,electrode_count,walking_condition_count,trig_count,:] = clean_segment
                    
                else:
                    all_clean_segments[subject,electrode_count,walking_condition_count,trig_count,:] = segment
                   # clean_segment_matrix[trig_count,:] = segment # if there are not enough template segments, just use the original segment
    
                trig_count += 1
                 
                ## plots of individual segments
                # plt.figure() 
                # plt.plot(average_temp_segments,'k')
                # plt.plot(segment,'b')
                # plt.title(len(segment_locs))
                # plt.plot(clean_segment,'g')
                
            walking_condition_count += 1   
            # clean_SSVEP = clean_segment_matrix[0:trig_count,:].mean(axis=0)
            # clean_SSVEP = clean_SSVEP - clean_SSVEP.mean()
            
          
            
        electrode_count += 1
          
          
#plt.legend()

np.save(path + 'all_clean_segments',all_clean_segments)