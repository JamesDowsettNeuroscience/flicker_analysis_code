#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 10:50:27 2023

@author: James Dowsett
"""

## analyse eye tracking flicker study, no people condition only



############ analysis pipeline for VOR-smooth pursuit ficker experiment ##################

from flicker_analysis_package import functions

import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

import mne

### information about the experiment:

path = '/media/james/USB DISK/eye_tracker_shutter_glasses/people_experiment_1/' # put path here



frequency_names = ('10 Hz', '40 Hz')

sample_rate = 5000

num_subjects = 10

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




for frequency in (10,40):

    decoding_scores = np.zeros([64,10])

    for subject in range(1,11):
        
            
        # plt.figure()
        # plt.suptitle('Subject ' + str(subject))
        
        
        print('\n Subject ' + str(subject) + '\n')
        
        
        
        if frequency == 10:
            period = period_10Hz
        if frequency == 40:
            period = period_40Hz    
        
        ### load data
        
        file_name =  'subject_' + str(subject) + '/eye_tracking_S' + str(subject) + '_' + str(frequency) + 'Hz_4'
        
        print('\n' + file_name + '\n')
        
              
        print('Loading electrode data ...')
        print('  ')
        
        # read the EEG data with the MNE function
        raw = mne.io.read_raw_brainvision(path + file_name + '.vhdr', preload=True)
    
    
        
        
        
        ## first save the EOG channels seperatly, because they will be interpolated
        HEOG_data = np.array(raw[31,:], dtype=object) 
        HEOG_data = HEOG_data[0,]
        HEOG_data = HEOG_data.flatten()
        
        
        # low pass HEOG
        nyquist = sample_rate/2
        cutoff = 1 # cutoff in Hz
        Wn = cutoff/nyquist
        b, a = scipy.signal.butter(3, Wn, 'low')
        HEOG_data = scipy.signal.filtfilt(b, a, HEOG_data)
        
        HEOG_data = HEOG_data - HEOG_data.mean() # baseline correct
    
    
    
        ### laplacian 
    
    
        ### set the montage
     
        raw.info.ch_names[60] = 'Fpz' # correct the channel named incorrectly
     
        # raw.info.ch_names.remove('trigger')
        # raw.info.ch_names.remove('HEOG')
    
        # rename the trigger and EOG as the position of the ground and reference, to be interpolated
        raw.info.ch_names[30] =  'FCz' #'A1' 
        raw.info.ch_names[31] = 'AFz' # 'A2'  #
     
         # The EEG channels use the standard naming strategy.
         # By supplying the 'montage' parameter, approximate locations
         # will be added for them
     
        montage = 'standard_1005'
     
        sfreq = 5000  # sample rate in Hertz
        # # Initialize required fields
        ch_types = ['eeg'] * 30 + ['misc'] * 2 + ['eeg'] * 32
        raw.info = mne.create_info(raw.info.ch_names, sfreq, ch_types = 'eeg')
     
    
     #raw.info.set_montage(montage)
        raw.set_montage(montage)
    
        
        raw.info['bads']  = [] # reset bad channels to an empty list
        
        bad_electrodes = []
        
        # include position of reference and ground in channels to be interpolated: use trigger and EOG channels
        bad_electrodes.append('AFz')
        bad_electrodes.append('FCz')
        
        # mark the bad electrodes as bad in MNE
        raw.info['bads'] = bad_electrodes
        
            #### interpolate bad channels
        
        eeg_data_interp = raw.copy().interpolate_bads(reset_bads=True) # must reset bad channels for next processing steps
            
        print('Computing Laplacian ...')
        
        raw_csd = mne.preprocessing.compute_current_source_density(eeg_data_interp)  
    
    
    
    
        ###### put data into numpy matrix
    
        all_data = np.zeros([64,len(raw)])
    
        for chan in range(0,64):
    
            data = np.array(raw_csd[chan,:], dtype=object) 
           # data = np.array(raw[chan,:], dtype=object) 
            data = data[0,]
            data = data.flatten()
        
            all_data[chan,:] = data
     
    
    
    
    
    
    
        
        ### read triggers
        
        f = open(path + file_name + '.vmrk') # open the .vmrk file, call it "f"
        
        # use readline() to read the first line 
        line = f.readline()
    
        all_triggers_list = []
        
        flicker_trigger_name_1 = 'S  4'
        flicker_trigger_name_2 = 'S128'
        
        
        while line:
    
            if flicker_trigger_name_1 in line: # if the line contains a flicker trigger
                
                # get the trigger time from line
                start = line.find(flicker_trigger_name_1 + ',') + len(flicker_trigger_name_1) + 1
                end = line.find(",1,")
                trigger_time = line[start:end]       
               
                # append the trigger time to the correct condition
                
                all_triggers_list.append(trigger_time)
                
            if flicker_trigger_name_2 in line: # if the line contains a flicker trigger
                
                # get the trigger time from line
                start = line.find(flicker_trigger_name_2 + ',') + len(flicker_trigger_name_2) + 1
                end = line.find(",1,")
                trigger_time = line[start:end]       
               
                # append the trigger time to the correct condition
                
                all_triggers_list.append(trigger_time)
                  
                
                
            
            line = f.readline() # use realine() to read next line
            line
            
        f.close() # close the file
        
    
         # convert to numpy arrays
        all_triggers = np.array(all_triggers_list, dtype=int)
         
        print(str(len(all_triggers)) + ' triggers found')
         
        print('Average diff = ' + str(np.diff(all_triggers).mean()))
        
        
         # make trigger time series
        trigger_time_series = np.zeros([len(HEOG_data),])
        
         
        for trigger in all_triggers:
            trigger_time_series[trigger] = 0.0001
        
        
       # plt.plot(trigger_time_series)
        
        
        
        ## sort triggers into eyes left and eyes right
        
        eyes_right_triggers_list = []
        eyes_left_triggers_list = []
        
        eyes_right_triggers_time_series = np.zeros([len(HEOG_data),])
        eyes_left_triggers_time_series = np.zeros([len(HEOG_data),]) 
        
        for trigger in all_triggers:
            
            if HEOG_data[trigger] > 0:
                eyes_right_triggers_list.append(trigger)
                eyes_right_triggers_time_series[trigger] = 0.0001
            
            if HEOG_data[trigger] < 0:
                eyes_left_triggers_list.append(trigger)
                eyes_left_triggers_time_series[trigger] = 0.0001
            
            
        # convert to array
        eyes_right_triggers = np.asarray(eyes_right_triggers_list)
        eyes_left_triggers = np.asarray(eyes_left_triggers_list)
        
        
        # plt.plot(eyes_right_triggers_time_series)
        # plt.plot(eyes_left_triggers_time_series)
        
        # plt.plot(HEOG_data)
        
        
        
         
        #### make SSVEPs   ####################
        
    
        # plt_count = 0
        # for electrode in (6,7):  #range(0,64):
        
        #   data = all_data[electrode,:]
        
        #   plt_count += 1
        #   plt.subplot(1,2,plt_count)
        #   plt.title(raw.ch_names[electrode])
        
        #   for eye_position in ('Right', 'Left'):
              
        #       print(eye_position)
        
        #       if eye_position == 'Right':
        #           triggers = eyes_right_triggers
        #       elif eye_position == 'Left':
        #           triggers = eyes_left_triggers
        
        #         # ### linear interpolation
        #       if frequency == 10:
        #           trig_1_time = trig_1_time_10Hz
        #           trig_2_time = trig_2_time_10Hz
        #           trig_length =  trig_length_10Hz
        #       elif frequency == 40:
        #           trig_1_time = trig_1_time_40Hz
        #           trig_2_time = trig_2_time_40Hz
        #           trig_length =  trig_length_40Hz
            
            
        #       data_linear_interpolation = functions.linear_interpolation(data, triggers, trig_1_time, trig_2_time, trig_length)
            
    
        #       # make SSVEP
        #       SSVEP = functions.make_SSVEPs(data_linear_interpolation, triggers, period) # 
    
        #       SSVEP = SSVEP - SSVEP.mean()
              
        #       plt.plot(SSVEP)
        
        
        ##########    Decode eye position       ##############
    
    
                # ### setup linear interpolation
        if frequency == 10:
            trig_1_time = trig_1_time_10Hz
            trig_2_time = trig_2_time_10Hz
            trig_length =  trig_length_10Hz
        elif frequency == 40:
            trig_1_time = trig_1_time_40Hz
            trig_2_time = trig_2_time_40Hz
            trig_length =  trig_length_40Hz
      
        
        import random 
        


            
            
        for electrode in range(0,64):
            
            print(electrode)
        
            data = all_data[electrode,:] # load data
            
            
            num_loops = 10
            scores_right = np.zeros([num_loops,])
            scores_left = np.zeros([num_loops,])
            
            for loop in range(0,num_loops):     
            
                ## split triggers into training and test with a 50/50 split
                
                # first right
                
                num_eyes_right_triggers = len(eyes_right_triggers)
                
                seg_nums = np.arange(0,num_eyes_right_triggers) # an index for seach segment
             
                random.shuffle(seg_nums) # randomize the order
                
                training_trig_nums = seg_nums[0:int(num_eyes_right_triggers/2)]
                test_trig_nums = seg_nums[int(num_eyes_right_triggers/2):num_eyes_right_triggers]
                
                training_eyes_right_triggers = eyes_right_triggers[training_trig_nums]
                
                test_eyes_right_triggers = eyes_right_triggers[test_trig_nums]
                
               
                # then left
                
                num_eyes_left_triggers = len(eyes_left_triggers)
                
                seg_nums = np.arange(0,num_eyes_left_triggers) # an index for seach segment
             
                random.shuffle(seg_nums) # randomize the order
                
                training_trig_nums = seg_nums[0:int(num_eyes_left_triggers/2)]
                test_trig_nums = seg_nums[int(num_eyes_left_triggers/2):num_eyes_left_triggers]
                
                training_eyes_left_triggers = eyes_left_triggers[training_trig_nums]
                
                test_eyes_left_triggers = eyes_left_triggers[test_trig_nums]    
                    
                
        
                ### make training SSVEPs
                
                data_linear_interpolation = functions.linear_interpolation(data, training_eyes_right_triggers, trig_1_time, trig_2_time, trig_length)
                
                training_SSVEP_eyes_right = functions.make_SSVEPs(data_linear_interpolation, training_eyes_right_triggers, period) # 
                
                data_linear_interpolation = functions.linear_interpolation(data, training_eyes_left_triggers, trig_1_time, trig_2_time, trig_length)
                
                training_SSVEP_eyes_left = functions.make_SSVEPs(data_linear_interpolation, training_eyes_left_triggers, period) # 
                
            
                # plt.plot(training_SSVEP_eyes_right,'r')
                # plt.plot(training_SSVEP_eyes_left,'g')
            
            
            #  make test SSVEPs
                
                data_linear_interpolation = functions.linear_interpolation(data, test_eyes_right_triggers, trig_1_time, trig_2_time, trig_length)
                
                test_SSVEP_eyes_right = functions.make_SSVEPs(data_linear_interpolation, test_eyes_right_triggers, period) # 
                
                data_linear_interpolation = functions.linear_interpolation(data, test_eyes_left_triggers, trig_1_time, trig_2_time, trig_length)
                
                test_SSVEP_eyes_left = functions.make_SSVEPs(data_linear_interpolation, test_eyes_left_triggers, period) # 
                
                
                # plt.plot(test_SSVEP_eyes_right,'r')
                # plt.plot(test_SSVEP_eyes_left,'g')
                
                
                ## test eyes right decoding
                eyes_right_corr = np.corrcoef(training_SSVEP_eyes_right,test_SSVEP_eyes_right)[0,1]
                eyes_left_corr = np.corrcoef(training_SSVEP_eyes_left,test_SSVEP_eyes_right)[0,1]
                
                if eyes_right_corr > eyes_left_corr:
                    scores_right[loop] = 1
                    
                    
                ## test eyes left decoding
                eyes_right_corr = np.corrcoef(training_SSVEP_eyes_right,test_SSVEP_eyes_left)[0,1]
                eyes_left_corr = np.corrcoef(training_SSVEP_eyes_left,test_SSVEP_eyes_left)[0,1]
                
                if eyes_left_corr > eyes_right_corr:
                    scores_left[loop] = 1
                   
                
                
            percent_correct_right = np.sum(scores_right) * (100/num_loops)
            percent_correct_left = np.sum(scores_left) * (100/num_loops)
           
            average_percent_correct = (percent_correct_right + percent_correct_left) / 2
            
            decoding_scores[electrode, subject-1] = average_percent_correct
            
            
            
            
    grand_average_decoding_scores = decoding_scores.mean(axis=1)
            
    
    for electrode in range(0,64):
        
        print(raw.ch_names[electrode] + '  ' + str(grand_average_decoding_scores[electrode])) 
        
    
        
    
    #### Topo plot
    
    fig = plt.figure()
    
    plt.title(str(frequency) + ' Hz')
    
    min_value = 50
    max_value = 100
    
    # Initialize required fields
    
    channel_names = raw.ch_names
    
    info = mne.create_info(channel_names, sfreq, ch_types = 'eeg')
    
    info.set_montage(montage)
                
      
    values_to_plot = grand_average_decoding_scores
      
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
    

