#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 12:38:48 2023

@author: James Dowsett
"""

############ analysis pipeline for VOR-smooth pursuit ficker experiment ##################

from flicker_analysis_package import functions

import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

import random

### information about the experiment:

#path = '/home/james/Active_projects/VOR_EEG/' # put path here
path = '/media/james/Expansion/VOR_EEG/'

with open(path + "chan_names", "rb") as fp:   # Unpickling electrode names
    electrode_names = pickle.load(fp)

condition_names = ('Eyes Track, head fixed', 'Head track, Eyes fixed', 'Both head and eyes track', 'Eyes Track, head fixed - control', 'Head track, Eyes fixed - control', 'Both head and eyes track - control')

frequency_names = ('10 Hz', '40 Hz')

sample_rate = 5000

num_subjects = 17

period_10Hz = 500 # period of flicker in data points
period_40Hz = 125 # period of flicker in data points

length = 1 # length of FFT in seconds

## trigger times for linear interpolation
trig_1_time_10Hz = -1
trig_2_time_10Hz = 248
trig_length_10Hz = 20

trig_1_time_40Hz = -1
trig_2_time_40Hz = 59
trig_length_40Hz = 20


#######################################

laplacian = 1 # 0 = FCz reference, 1 = laplacian re-reference



for subject in range(1,num_subjects+1):
    
    print('\n Subject ' + str(subject) + '\n')
    

    for frequency in (0,1): # 0 = 10Hz, 1 = 40Hz
    
        # plt.figure()
        # if laplacian == 0:
        #     plt.suptitle('Subject ' + str(subject) + ' ' + frequency_names[frequency])
        # elif laplacian == 1:
        #     plt.suptitle('Subject ' + str(subject) + ' ' + frequency_names[frequency] + ' Laplacian')
    
    
        if frequency == 0:
            print('\n 10 Hz \n')
        elif frequency == 1:
            print('\n 40 Hz \n')
            
            
        
        for condition_count in range(0,6):
            
            #### figures for plotting individual SSVEPs
            
            # if condition_count == 0 or condition_count == 3:
            #     plt.figure((subject*10)+ (frequency*50) + 0)
            #     plt.suptitle('Subject ' + str(subject) + ' ' + frequency_names[frequency] + '  ' + condition_names[0])
            # elif condition_count == 1 or condition_count == 4:
            #     plt.figure((subject*10)+ (frequency*50) + 1)
            #     plt.suptitle('Subject ' + str(subject) + ' ' + frequency_names[frequency] + '  ' + condition_names[1])
            # elif condition_count == 2 or condition_count == 5:
            #     plt.figure((subject*10)+ (frequency*50) + 2)
            #     plt.suptitle('Subject ' + str(subject) + ' ' + frequency_names[frequency] + '  ' + condition_names[2])
                
                
            
            
            if frequency == 0:
                condition = condition_count+1
                period = period_10Hz
            elif frequency == 1:
                condition = condition_count+7 
                period = period_40Hz
                
            print(condition_names[condition_count])
            
        
            ## load all data
            # if laplacian == 0:
            #     all_data = np.load(path + 'S' + str(subject) + '_' + str(condition) + '_all_data.npy')
            # elif laplacian == 1:
            #     all_data = np.load(path + 'S' + str(subject) + '_' + str(condition) + '_all_data_laplacian.npy')
            
            ## load EOG data
            HEOG_data = np.load(path +  'S' + str(subject) + '_' + str(condition) + '_HEOG.npy')
            VEOG_data = np.load(path +  'S' + str(subject) + '_' + str(condition) + '_VEOG.npy')
        
        
            HEOG_data = HEOG_data - HEOG_data.mean() # baseline correct
            VEOG_data = VEOG_data - VEOG_data.mean()
            
            
            ## filter EOG data
            
            nyquist = sample_rate/2
        
            
            # low pass HEOG
            cutoff = 1 # cutoff in Hz
            Wn = cutoff/nyquist
            b, a = scipy.signal.butter(3, Wn, 'low')
            HEOG_data = scipy.signal.filtfilt(b, a, HEOG_data)
           
            if condition_count < 3: # only high pass filter the eyes moving conditions 
                # high pass HEOG
                cutoff = 0.05 # cutoff in Hz
                Wn = cutoff/nyquist
                b, a = scipy.signal.butter(3, Wn, 'high')
                HEOG_data = scipy.signal.filtfilt(b, a, HEOG_data)
        
        
        
            # low pass VEOG
            cutoff = 5 # cutoff in Hz
            Wn = cutoff/nyquist
            b, a = scipy.signal.butter(3, Wn, 'low')
            VEOG_data = scipy.signal.filtfilt(b, a, VEOG_data)
            
            # high pass VEOG
            cutoff = 1 # cutoff in Hz
            Wn = cutoff/nyquist
            b, a = scipy.signal.butter(3, Wn, 'high')
            VEOG_data = scipy.signal.filtfilt(b, a, VEOG_data)
        
            
        
            ## load triggers
            all_flicker_triggers = np.load(path + 'S' + str(subject) + '_' + str(condition) + '_flicker_triggers.npy')
            
            all_movement_triggers = np.load(path + 'S' + str(subject) + '_' + str(condition) + '_movement_triggers.npy')
            
            # make trigger time series
            flicker_trigger_time_series = np.zeros([all_flicker_triggers[-1]+1,])
            movement_trigger_time_series = np.zeros([all_movement_triggers[-1]+1,])
            
            for trigger in all_flicker_triggers:
                flicker_trigger_time_series[trigger] = 0.0001
                
            for trigger in all_movement_triggers:
                movement_trigger_time_series[trigger] = 0.0002
               
           #  plt.figure()
           #  plt.title(condition_names[condition-1])
           # # plt.plot(flicker_trigger_time_series)
           #  plt.plot(movement_trigger_time_series)
            
        
            
            ## sort triggers 
            
            print('Sorting triggers ...')
            
            good_flicker_triggers_list = []
            
            
            wait_time = 5000 # one second wait time for the moving condition
        
            for trigger in all_flicker_triggers:
                
                trigger_distance_from_movement = all_movement_triggers - trigger
                
                # for the movement condition, the trigger marks the start of the movement, preceded by 0ne second of still
                if condition_count < 3: 
                    
                    # select the next movement trigger in the future
                    if len(trigger_distance_from_movement[trigger_distance_from_movement>0]) > 0:
                        nearest_movement_trigger = min(trigger_distance_from_movement[trigger_distance_from_movement>0])
                        
                        # if next movement trigger is more than one second away
                        if nearest_movement_trigger > wait_time: 
                            
                            # check for eye blinks
                            VEOG_segment = VEOG_data[trigger-500:trigger+500]
                            
                            if np.ptp(VEOG_segment) < 0.00005:
                            
                                good_flicker_triggers_list.append(trigger)
                                
                                
                   # for the fixation conditions, the trigger marks start of approx. 1 second of movement         
                elif condition_count >=3: 
                    
                    # select the next movement trigger in the past
                    if len(trigger_distance_from_movement[trigger_distance_from_movement<0]) > 0:
                       
                        nearest_movement_trigger = max(trigger_distance_from_movement[trigger_distance_from_movement<0])
                    
                        if np.abs(nearest_movement_trigger) > 5000: # ignore triggers 1 second after movement trigger
                        
                            # check for eye blinks
                            VEOG_segment = VEOG_data[trigger-500:trigger+500]
                            
                            if np.ptp(VEOG_segment) < 0.00005:
                            
                                good_flicker_triggers_list.append(trigger)
                    
                    else: # include all triggers before the first movement trigger
                        good_flicker_triggers_list.append(trigger)
                
               
            # convert to array
            good_flicker_triggers = np.asarray(good_flicker_triggers_list)   
            
            ## save triggers
            if frequency == 0:
                triggers_file_name = 'S' + str(subject) + '_' + str(condition_count) + '_10Hz_good_triggers'
            if frequency == 1:
                triggers_file_name = 'S' + str(subject) + '_' + str(condition_count) + '_40Hz_good_triggers'          
               
                
            np.save(path + triggers_file_name, good_flicker_triggers)
                
            # plot time series of good triggers   
            # good_flicker_triggers_time_series = np.zeros([all_flicker_triggers[-1]+1,])
            # for trigger in good_flicker_triggers_list:
            #     good_flicker_triggers_time_series[trigger] = 0.00015
             
                
            # plt.plot(good_flicker_triggers_time_series)
            
            # plt.plot(HEOG_data)
            # plt.plot(VEOG_data)
            
            
            ######### sort triggers into eyes left and eyes right ###############################

    #         print('Sorting triggers into left and right ...')

    # # put all triggers into one array
    #         good_flicker_triggers_array = np.asarray(good_flicker_triggers_list)
    #         all_triggers_combined = np.concatenate((all_movement_triggers, good_flicker_triggers_array), axis=0)
    #         all_triggers_combined = np.sort(all_triggers_combined) # sort into order


    #         eyes_right_triggers_list = []
    #         eyes_left_triggers_list = []
            
    #         # right_15_triggers = []
    #         # right_10_triggers = []
    #         # right_5_triggers = []
    #         # centre_triggers = []
    #         # left_5_triggers = []
    #         # left_10_triggers = []
    #         # left_15_triggers = []

    #         movement_direction = 'start'  # ignore first movement trigger is starting to move to the right            

    #         if condition_count <= 2: # movement conditions

    #             for t in all_triggers_combined:
          
                    
    #                 if t in all_movement_triggers:
    #                     # switch direction
    #                     if movement_direction == 'left' or movement_direction == 'start':
    #                         movement_direction = 'right'
    #                     elif movement_direction == 'right':
    #                         movement_direction = 'left'
                         
    #                     last_movement_trigger = np.copy(t)
    #                   #  print(last_movement_trigger)
                
    #                 if t in good_flicker_triggers_list:
                        
    #                     time_from_last_movement_trigger = t - last_movement_trigger
            
    #                     if time_from_last_movement_trigger < 12500 and movement_direction == 'right':
    #                         eyes_left_triggers_list.append(t)
    #                     elif time_from_last_movement_trigger > 12500 and movement_direction == 'right':
    #                         eyes_right_triggers_list.append(t)
    #                     elif time_from_last_movement_trigger < 12500 and movement_direction == 'left':
    #                         eyes_right_triggers_list.append(t)
    #                     elif time_from_last_movement_trigger > 12500 and movement_direction == 'left':
    #                         eyes_left_triggers_list.append(t)
            
    #         elif condition_count > 2: # control conditions
                
             
    #             arm_positions = [10, 15, 10, 5, 0, -5, -10, -15, -10, -5, 0, 5, 10, 15, 10, 5, 0, -5, -10, -15]
                
    #             index = 0
                
    #             for t in all_triggers_combined:
                    
    #                 if t in all_movement_triggers:
    #                     index = int(np.where(all_movement_triggers == t)[0])
                    
                 
    #                 if t in good_flicker_triggers_list:
                        
    #                     if arm_positions[index] > 0:
    #                         eyes_right_triggers_list.append(t)
    #                     elif arm_positions[index] < 0:
    #                         eyes_left_triggers_list.append(t)
                       
    #                                 # seperate triggers for each arm position
                                    
    #                     # if arm_positions[index] == 15:
    #                     #     right_15_triggers.append(t)
    #                     # elif arm_positions[index] == 10:
    #                     #     right_10_triggers.append(t)
    #                     # elif arm_positions[index] == 5:
    #                     #     right_5_triggers.append(t)
    #                     # elif arm_positions[index] == 0:
    #                     #     centre_triggers.append(t)
    #                     # elif arm_positions[index] == -5:
    #                     #     left_5_triggers.append(t)
    #                     # elif arm_positions[index] == -10:
    #                     #     left_10_triggers.append(t)
    #                     # elif arm_positions[index] == -15:
    #                     #     left_15_triggers.append(t)                            

 


    #        ###########################################################

    #         # convert to array
    #         eyes_right_triggers = np.asarray(eyes_right_triggers_list)
    #         eyes_left_triggers = np.asarray(eyes_left_triggers_list)
            
                 
            
            # ## make trigger time series
            # eyes_right_time_series = np.zeros([all_flicker_triggers[-1]+1,])
            
            # for trigger in eyes_right_triggers:
            #     eyes_right_time_series[trigger] = 0.00017
            
            # eyes_left_time_series = np.zeros([all_flicker_triggers[-1]+1,])
            
            # for trigger in eyes_left_triggers:
            #     eyes_left_time_series[trigger] = 0.00018
                
            # plt.plot(eyes_right_time_series)
            # plt.plot(eyes_left_time_series)
            
            
            
            
            ########### make SSVEPs for each electrode  ################
                  
            # # values for y axis scaling
            # max_value = 0
            # min_value = 0
              
                
            # for target_position in (0,1): # in range(0,7):  # 
                

            #     if target_position == 0:
            #         triggers = eyes_right_triggers
            #     elif target_position == 1:
            #         triggers = eyes_left_triggers
                    
            #     # if target_position == 0:
            #     #     triggers = right_15_triggers
            #     # elif target_position == 1:
            #     #     triggers = right_10_triggers
            #     # elif target_position == 2:
            #     #     triggers = right_5_triggers
            #     # elif target_position == 3:
            #     #     triggers = centre_triggers
            #     # elif target_position == 4:
            #     #     triggers = left_5_triggers
            #     # elif target_position == 5:
            #     #     triggers = left_10_triggers
            #     # elif target_position == 6:
            #     #     triggers = left_15_triggers
                    
                    
      
     
                
            #     plt_count = 0
            #     for electrode in (6,7):  #range(0,64):
                    
            #         plt_count += 1
                    
            #         data = all_data[electrode,:]
            
            #         # ### linear interpolation
            #         if frequency == 0:
            #             trig_1_time = trig_1_time_10Hz
            #             trig_2_time = trig_2_time_10Hz
            #             trig_length =  trig_length_10Hz
            #         elif frequency == 1:
            #             trig_1_time = trig_1_time_40Hz
            #             trig_2_time = trig_2_time_40Hz
            #             trig_length =  trig_length_40Hz
    
                    
            #         data_linear_interpolation = functions.linear_interpolation(data, triggers, trig_1_time, trig_2_time, trig_length)
                    
            
            #         # make SSVEP
            #         SSVEP = functions.make_SSVEPs(data_linear_interpolation, triggers, period) # SIGI was always 40 Hz, length = 25
            
            
            #         #plt.subplot(8,8,plt_count)
            #         plt.subplot(1,2,plt_count)
                    
            #         if target_position == 0:
            #             plt.plot(SSVEP, label = condition_names[condition_count] + ' right')
            #         elif target_position == 1:
            #             plt.plot(SSVEP, label = condition_names[condition_count] + ' left')
                    
            #         # plt.plot(SSVEP, label = target_position)
                    
            #         plt.title(electrode_names[electrode] + '  ' + frequency_names[frequency])
                
            #         if max(SSVEP) > max_value:
            #             max_value = max(SSVEP)
            #         if min(SSVEP) < min_value:
            #             min_value = min(SSVEP)
                        
                        
            # # adjust y axis      
            # plt_count = 0   
            # for electrode in (6,7):  #range(0,64):    
            #     plt_count += 1
            #     #plt.subplot(8,8,plt_count)
            #     plt.subplot(1,2,plt_count)
            #     plt.ylim(min_value, max_value)
                
                
            # plt.legend()
            
            
            
            
            
            ##########    Decode eye position       ##############
        
        
            #         # ### setup linear interpolation
            # if frequency == 0:
            #     trig_1_time = trig_1_time_10Hz
            #     trig_2_time = trig_2_time_10Hz
            #     trig_length =  trig_length_10Hz
            # elif frequency == 1:
            #     trig_1_time = trig_1_time_40Hz
            #     trig_2_time = trig_2_time_40Hz
            #     trig_length =  trig_length_40Hz
          
            
            # import random 
            
    
                
            # for electrode in range(0,64):
                
            #     print(electrode)
            
            #     data = all_data[electrode,:] # load data
                
                
            #     num_loops = 10
            #     scores_right = np.zeros([num_loops,])
            #     scores_left = np.zeros([num_loops,])
                
            #     for loop in range(0,num_loops):     
                
            #         ## split triggers into training and test with a 50/50 split
                    
            #         # first right
                    
            #         num_eyes_right_triggers = len(eyes_right_triggers)
                    
            #         seg_nums = np.arange(0,num_eyes_right_triggers) # an index for seach segment
                 
            #         random.shuffle(seg_nums) # randomize the order
                    
            #         training_trig_nums = seg_nums[0:int(num_eyes_right_triggers/2)]
            #         test_trig_nums = seg_nums[int(num_eyes_right_triggers/2):num_eyes_right_triggers]
                    
            #         training_eyes_right_triggers = eyes_right_triggers[training_trig_nums]
                    
            #         test_eyes_right_triggers = eyes_right_triggers[test_trig_nums]
                    
                   
            #         # then left
                    
            #         num_eyes_left_triggers = len(eyes_left_triggers)
                    
            #         seg_nums = np.arange(0,num_eyes_left_triggers) # an index for seach segment
                 
            #         random.shuffle(seg_nums) # randomize the order
                    
            #         training_trig_nums = seg_nums[0:int(num_eyes_left_triggers/2)]
            #         test_trig_nums = seg_nums[int(num_eyes_left_triggers/2):num_eyes_left_triggers]
                    
            #         training_eyes_left_triggers = eyes_left_triggers[training_trig_nums]
                    
            #         test_eyes_left_triggers = eyes_left_triggers[test_trig_nums]    
                        
                    
            
            #         ### make training SSVEPs
                    
            #         data_linear_interpolation = functions.linear_interpolation(data, training_eyes_right_triggers, trig_1_time, trig_2_time, trig_length)
                    
            #         training_SSVEP_eyes_right = functions.make_SSVEPs(data_linear_interpolation, training_eyes_right_triggers, period) # 
                    
            #         data_linear_interpolation = functions.linear_interpolation(data, training_eyes_left_triggers, trig_1_time, trig_2_time, trig_length)
                    
            #         training_SSVEP_eyes_left = functions.make_SSVEPs(data_linear_interpolation, training_eyes_left_triggers, period) # 
                    
                
            #         # plt.plot(training_SSVEP_eyes_right,'r')
            #         # plt.plot(training_SSVEP_eyes_left,'g')
                
                
            #     #  make test SSVEPs
                    
            #         data_linear_interpolation = functions.linear_interpolation(data, test_eyes_right_triggers, trig_1_time, trig_2_time, trig_length)
                    
            #         test_SSVEP_eyes_right = functions.make_SSVEPs(data_linear_interpolation, test_eyes_right_triggers, period) # 
                    
            #         data_linear_interpolation = functions.linear_interpolation(data, test_eyes_left_triggers, trig_1_time, trig_2_time, trig_length)
                    
            #         test_SSVEP_eyes_left = functions.make_SSVEPs(data_linear_interpolation, test_eyes_left_triggers, period) # 
                    
                    
            #         # plt.plot(test_SSVEP_eyes_right,'r')
            #         # plt.plot(test_SSVEP_eyes_left,'g')
                    
                    
            #         ## test eyes right decoding
            #         eyes_right_corr = np.corrcoef(training_SSVEP_eyes_right,test_SSVEP_eyes_right)[0,1]
            #         eyes_left_corr = np.corrcoef(training_SSVEP_eyes_left,test_SSVEP_eyes_right)[0,1]
                    
            #         if eyes_right_corr > eyes_left_corr:
            #             scores_right[loop] = 1
                        
                        
            #         ## test eyes left decoding
            #         eyes_right_corr = np.corrcoef(training_SSVEP_eyes_right,test_SSVEP_eyes_left)[0,1]
            #         eyes_left_corr = np.corrcoef(training_SSVEP_eyes_left,test_SSVEP_eyes_left)[0,1]
                    
            #         if eyes_left_corr > eyes_right_corr:
            #             scores_left[loop] = 1
                       
                    
                    
            #     percent_correct_right = np.sum(scores_right) * (100/num_loops)
            #     percent_correct_left = np.sum(scores_left) * (100/num_loops)
               
            #     average_percent_correct = (percent_correct_right + percent_correct_left) / 2
                
            #     decoding_scores[electrode, frequency, condition_count, subject-1] = average_percent_correct
                
             
                
             
#### make example SSVEPs from Pz ###########

electrode = 18 # 18 = Pz
           
for subject in range(1,num_subjects+1):
    
    plt.figure()
    plt.suptitle('Subject ' + str(subject))
    
    print('\n Subject ' + str(subject) + '\n')
    
    for frequency in range(0,2):
        
        print('\n ' + frequency_names[frequency])
        

        # ### linear interpolation times
        if frequency == 0:
            trig_1_time = trig_1_time_10Hz
            trig_2_time = trig_2_time_10Hz
            trig_length =  trig_length_10Hz
        elif frequency == 1:
            trig_1_time = trig_1_time_40Hz
            trig_2_time = trig_2_time_40Hz
            trig_length =  trig_length_40Hz
        
        
        for condition_type in range(0,3):
            
            print(condition_names[condition_type])
            
            plt.subplot(2,3,(frequency*3) + (condition_type+1))
            plt.title(frequency_names[frequency] + ' ' + condition_names[condition_type])
            
            if condition_type == 0:
                conditions_to_use = (0,3)
            elif condition_type == 1:
                conditions_to_use = (1,4)
            elif condition_type == 2:
                conditions_to_use = (2,5)
                        
                
            # get data and triggers for active and control conditions and save each seperatly
            for active_or_control in range(0,2):
                
                condition_count = conditions_to_use[active_or_control]
        

                # get correct values for file name
                if frequency == 0:
                    condition = condition_count+1
                    period = period_10Hz
                elif frequency == 1:
                    condition = condition_count+7 
                    period = period_40Hz
                 
                ## load data    
                if laplacian == 0:
                    all_data = np.load(path + 'S' + str(subject) + '_' + str(condition) + '_all_data.npy')
                elif laplacian == 1:
                    all_data = np.load(path + 'S' + str(subject) + '_' + str(condition) + '_all_data_laplacian.npy')      
                 
                ## load triggers
                
                if frequency == 0:
                    triggers = np.load(path + 'S' + str(subject) + '_' + str(condition_count) + '_10Hz_good_triggers.npy')
                if frequency == 1:
                    triggers = np.load(path + 'S' + str(subject) + '_' + str(condition_count) + '_40Hz_good_triggers.npy')
                   
                
                 
                if active_or_control == 0:
                    all_active_data = np.copy(all_data)
                    active_triggers = np.copy(triggers)
                elif active_or_control == 1:
                    all_control_data = np.copy(all_data)
                    control_triggers = np.copy(triggers)
                   
              
                ## get data from two conditions (active and control)

            active_data = all_active_data[electrode,:]
            control_data = all_control_data[electrode,:]
        
            # linear interpolation
            active_data = functions.linear_interpolation(active_data, active_triggers, trig_1_time, trig_2_time, trig_length)
            control_data = functions.linear_interpolation(control_data, control_triggers, trig_1_time, trig_2_time, trig_length)
        
            active_SSVEP = functions.make_SSVEPs(active_data, active_triggers, period)
            control_SSVEP = functions.make_SSVEPs(control_data, control_triggers, period)
         
            plt.plot(active_SSVEP, label = 'Moving')
            plt.plot(control_SSVEP, label = 'Still')
             
            plt.legend()
                
             
     
        ############# decode each condition vs. control   ##############     
 
subjects_to_use = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17]

decoding_scores = np.zeros([64, 2, 3, len(subjects_to_use)]) # electrode, frequency, condition type , subject   
 
subject_count = 0

for subject in subjects_to_use:
    
    print('\n Subject ' + str(subject) + '\n')
    
    for frequency in range(0,2):
        
        print('\n ' + frequency_names[frequency])
        
        
        
        # ### linear interpolation times
        if frequency == 0:
            trig_1_time = trig_1_time_10Hz
            trig_2_time = trig_2_time_10Hz
            trig_length =  trig_length_10Hz
        elif frequency == 1:
            trig_1_time = trig_1_time_40Hz
            trig_2_time = trig_2_time_40Hz
            trig_length =  trig_length_40Hz
        
        
        for condition_type in range(0,3):
            
            print(condition_names[condition_type])
            
            if condition_type == 0:
                conditions_to_use = (0,3)
            elif condition_type == 1:
                conditions_to_use = (1,4)
            elif condition_type == 2:
                conditions_to_use = (2,5)
                        
                
            # get data and triggers for active and control conditions and save each seperatly
            for active_or_control in range(0,2):
                
                condition_count = conditions_to_use[active_or_control]
        

                # get correct values for file name
                if frequency == 0:
                    condition = condition_count+1
                    period = period_10Hz
                elif frequency == 1:
                    condition = condition_count+7 
                    period = period_40Hz
                 
                ## load data    
                if laplacian == 0:
                    all_data = np.load(path + 'S' + str(subject) + '_' + str(condition) + '_all_data.npy')
                elif laplacian == 1:
                    all_data = np.load(path + 'S' + str(subject) + '_' + str(condition) + '_all_data_laplacian.npy')      
                 
                ## load triggers
                
                if frequency == 0:
                    triggers = np.load(path + 'S' + str(subject) + '_' + str(condition_count) + '_10Hz_good_triggers.npy')
                if frequency == 1:
                    triggers = np.load(path + 'S' + str(subject) + '_' + str(condition_count) + '_40Hz_good_triggers.npy')
                   
                
                 
                if active_or_control == 0:
                    all_active_data = np.copy(all_data)
                    active_triggers = np.copy(triggers)
                elif active_or_control == 1:
                    all_control_data = np.copy(all_data)
                    control_triggers = np.copy(triggers)
        
        
            for electrode in range(0,64):
                
                
                
                ## get data from two conditions (active and control)

                active_data = all_active_data[electrode,:]
                control_data = all_control_data[electrode,:]
            
                # linear interpolation
                active_data = functions.linear_interpolation(active_data, active_triggers, trig_1_time, trig_2_time, trig_length)
                control_data = functions.linear_interpolation(control_data, control_triggers, trig_1_time, trig_2_time, trig_length)
            
            
                ## Decode correlations
            
                num_loops = 10
                
                average_percent_correct = functions.decode_correlation(active_data, control_data, active_triggers, control_triggers, period, num_loops)
                
                print('\nElectrode number: ' + str(electrode) + ' -- ' + electrode_names[electrode] + ' \nAverage percent correct = ' +  str(average_percent_correct) + '\n')
                
                decoding_scores[electrode, frequency, condition_type, subject_count] = average_percent_correct
                

    subject_count += 1
    
    

### save decoding scores

np.save(path + 'decoding_scores_' + str(num_loops) + '_loops', decoding_scores)

### load decoding scores

decoding_scores = np.load(path + 'decoding_scores_' + str(num_loops) + '_loops.npy')




#### Topo plot

fig = plt.figure()

print('\nPz decoding accuracy: ')

for frequency in (0,1): # 0 = 10Hz, 1 = 40Hz
    
  #  plt.suptitle(frequency_names[frequency])
  
    print('\n' + frequency_names[frequency] + '\n')

    for condition_type in range(0,3):
        
        decoding_scores_for_condition = decoding_scores[:,frequency,condition_type,:]
        
        print(condition_names[condition_type] )
        print(decoding_scores_for_condition[18,:])
        print('Average decoding score = ' + str(decoding_scores_for_condition[18,:].mean()))
        print(' ')
        
        values_to_plot =  decoding_scores_for_condition.mean(axis=1)
        
        
        plt.subplot(2,3,(condition_type+1) + (frequency*3))
        
        plt.title(condition_names[condition_type] + ' ' + frequency_names[frequency])
        
        min_value = 50
        max_value = 100
        
        # Initialize required fields
        
        channel_names = electrode_names
        
        import mne
        
        sfreq = sample_rate
        
        info = mne.create_info(channel_names, sfreq, ch_types = 'eeg')
        
        montage = 'standard_1005'
        
        info.set_montage(montage)
                    
          
          
        evoked_values = mne.EvokedArray(np.reshape(values_to_plot, (64,1)), info)
        
        evoked_values.set_montage(montage)
        
        mne.viz.plot_topomap(evoked_values.data[:, 0], evoked_values.info,
        vmin=min_value, vmax=max_value, names=channel_names, show_names=True, show=True)
        
        im,cm = mne.viz.plot_topomap(evoked_values.data[:, 0], evoked_values.info,
        vmin=min_value, vmax=max_value, names=channel_names, show_names=True, show=True)
        
                                
        # manually fiddle the position of colorbar
        ax_x_start = 0.9
        ax_x_width = 0.04
        ax_y_start = 0.1
        ax_y_height = 0.8
        cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
        clb = fig.colorbar(im, cax=cbar_ax)
        
        
        
                
            
            
            
            
            