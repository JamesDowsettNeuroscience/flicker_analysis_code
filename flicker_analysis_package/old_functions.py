#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:28:37 2023

@author: James Dowsett
"""

#####  Old functions that i don't think i'll use any more, but i'll keep here just in case  ##############



### PRE-PROCESSING USING MNE


num_loops = 1000

## Function for loading data from one channel using electrode number
def load_data(file_name, electrode):
    
    import numpy as np
    import mne
    import os
    
    # read the EEG data with the MNE function
    raw = mne.io.read_raw_brainvision(file_name + '.vhdr')
    
    #channel_names = raw.info.ch_names
    
    # Load data
    print('  ')
    print('Loading Electrode data ...')
    print('  ')
    
    
    data = np.array(raw[electrode,:], dtype=object) 
    data = data[0,]
    data = data.flatten()


    print('Saving ...')

    electrode_data_file_name = file_name + '_' + raw.ch_names[electrode] +'_data'
    
    # save as a numpy array
    np.save(os.path.join(file_name, electrode_data_file_name), data)
    
    
    print('Done')
    
    

    
## Function for loading data from one channel using electrode name
def load_data_electrode_name(file_name, electrode_name):
    
    import numpy as np
    import mne
    import os
    
    # read the EEG data with the MNE function
    raw = mne.io.read_raw_brainvision(file_name + '.vhdr')
    
    channel_names = raw.info.ch_names
    
    electrode = channel_names.index(electrode_name)
    
    # Load data
    print('  ')
    print('Loading Electrode data ...')
    print('  ')
    
    data = np.array(raw[electrode,:], dtype=object) 
    data = data[0,]
    data = data.flatten()
    
    print('Saving ...')

    electrode_data_file_name = file_name + '_' + raw.ch_names[electrode] +'_data'
    
    # save as a numpy array
    np.save(os.path.join(file_name, electrode_data_file_name), data)
    
    print('Done')
    
    
    
    

### signal to noise ratio by randomly shuffling the data points of each segment and then making the SSVEP, compare to true SSVEP
## instead of peak to peak amplitude, use a time range around the peak

def SNR_random_peak_area(data, triggers, period):
    
    import numpy as np
    import random
    #import matplotlib.pyplot as plt
    
    segment_matrix = np.zeros([len(triggers), period]) # empty matrix to put segments into
    random_segment_matrix = np.zeros([len(triggers), period]) # empty matrix to put the randomly shuffled segments into
    seg_count = 0 # keep track of the number of segments
    
    # loop through all triggers and put the corresponding segment of data into the matrix
    for trigger in triggers:
        
        # select a segment of data the lenght of the flicker period, starting from the trigger time 
        segment =  np.copy(data[trigger:trigger+period]) 
        segment_matrix[seg_count,:] = np.copy(segment)
        
        random.shuffle(segment)

        random_segment_matrix[seg_count,:] = np.copy(segment)

        seg_count += 1
    
    true_SSVEP = segment_matrix.mean(axis=0) # average to make SSVEP
    random_SSVEP = random_segment_matrix.mean(axis=0) # average to make SSVEP of the randomly shuffled data

    true_SSVEP = true_SSVEP - true_SSVEP.mean() # baseline correct
    random_SSVEP = random_SSVEP - random_SSVEP.mean()

    for condition in ('true', 'random'):
        
        if condition == 'true':
            SSVEP = np.copy(true_SSVEP)
        elif condition == 'random':
            SSVEP = np.copy(random_SSVEP)

        SSVEP_repeated = np.tile(SSVEP, 2) # repeat the SSVEP in case the peak area is too near the beginning or end, loops around to the start  
    
        max_SSVEP_index = np.argmax(SSVEP) # get the index of the peak of the SSVEP
        
        ## get the average of the 5 data points around the peak of the SSVEP
        if (len(SSVEP) - max_SSVEP_index) <= 2: # if the peak is near the end of the SSVEP, 
            average_peak_area = SSVEP_repeated[max_SSVEP_index-2:max_SSVEP_index+3].mean()
        elif max_SSVEP_index <= 2: # if the peak index is near the begining of the SSVEP, repeat the SSVEP and move forward by the length of the SSVEP
            average_peak_area = SSVEP_repeated[max_SSVEP_index+len(SSVEP)-2:max_SSVEP_index+len(SSVEP)+3].mean()
        else: # otherwise, just average the area around the peak
            average_peak_area = SSVEP[max_SSVEP_index-2:max_SSVEP_index+3].mean()
        
        min_SSVEP_index = np.argmin(SSVEP) # get the index of the trough of the SSVEP
        
        ## get the average of the 5 data points around the trough of the SSVEP
        if (len(SSVEP) - min_SSVEP_index) <= 2: # if the trough is near the end of the SSVEP, 
            average_trough_area = SSVEP_repeated[min_SSVEP_index-2:min_SSVEP_index+3].mean()
        elif min_SSVEP_index <= 2: # if the trough index is near the begining of the SSVEP, move forward by the length of the SSVEP
            average_trough_area = SSVEP_repeated[min_SSVEP_index+len(SSVEP)-2:min_SSVEP_index+len(SSVEP)+3].mean()
        else: # otherwise, just average the area around the trough
            average_trough_area = SSVEP[min_SSVEP_index-2:min_SSVEP_index+3].mean()
    
    
        SSVEP_range = np.abs(average_peak_area - average_trough_area)

        if condition == 'true':
            true_SSVEP_range = np.copy(SSVEP_range)
        elif condition == 'random':
            random_SSVEP_range = np.copy(SSVEP_range)
    
    SNR = true_SSVEP_range/random_SSVEP_range
    
    return SNR


##Function to make SSVEPs with offset (use triggers & average)
def make_SSVEPs_offset(data, triggers, period, offset):
    import numpy as np

    segment_matrix = np.zeros([len(triggers), period]) # empty matrix to put segments into
    
    seg_count = 0 # keep track of the number of segments
    
    # loop through all triggers and put the corresponding segment of data into the matrix
    for trigger in triggers:
        
        # select a segment of data the lenght of the flicker period, starting from the trigger time 
        segment =  data[trigger-offset:trigger+period-offset] 
        
        segment_matrix[seg_count,:] = segment
    
        seg_count += 1
    
    SSVEP = segment_matrix.mean(axis=0) # average to make SSVEP
    
    SSVEP = SSVEP - SSVEP.mean() # baseline correct
    
    return SSVEP

 
##Function for only random SSVEPs and z score, with offset. Not necessary once trigger artefact is removed
def randomSSVEPs_zscore(SSVEP, data, all_triggers, period, num_loops, offset):
    
    import numpy as np
  #  import matplotlib.pyplot as plt
    import random
    
    random_amplitudes = np.zeros([num_loops,])
    
    for loop in range(0,num_loops):
        
        print(loop)
        # make random SSVEP 
        
        shuffled_segment_matrix =  np.zeros([len(all_triggers), period])  
        
        # loop through all triggers and put the corresponding segment of data into the matrix
        seg_count = 0 # keep track of the number of segments
        
        for trigger in all_triggers:
            
            segment =  data[trigger-offset:trigger+period-offset] 
        
            random.shuffle(segment) # randomly shuffle the data points
            
            shuffled_segment_matrix[seg_count,:] = segment
            
            seg_count += 1
        
        random_SSVEP = shuffled_segment_matrix.mean(axis=0) # average to make SSVEP
        
        random_SSVEP = random_SSVEP - random_SSVEP.mean() # baseline correct
        
        random_amplitudes[loop] = np.ptp(random_SSVEP)
    
    
  #  plt.plot(random_SSVEP,'k') # plot the last random shuffle, just to see
    
 #   plt.plot(SSVEP,'b') # plot the true SSVEP
    
    true_amplitude = np.ptp(SSVEP)
    
    print('True amplitude = ', true_amplitude)
    
    average_noise = random_amplitudes.mean()
    
    print('Amplitude noise = ', average_noise)
    
    std_noise = np.std(random_amplitudes)
    
    print('Standard Deviation noise = ', std_noise)
    
    Z_score  = (true_amplitude-average_noise) / std_noise
    
    print('Z_score = ', Z_score)
    
    

## Function for making SSVEP and removing walking/motion artefact. Works using adaptive template subtraction, 
## this requires a fairly predictable artefact such as consistent walking 

def make_SSVEP_artefact_removal(data, triggers, period, num_cycles_for_template, num_templates):

    import numpy as np
    from timeit import default_timer as timer
    from datetime import timedelta
    
    length_artefact_segment = period * num_cycles_for_template

    clean_segment_matrix = np.zeros([len(triggers),period])

    correlation_threshold = 0.8  # correlation between segment to be cleaned and template must be above this value to be considered

    k = 0
    trig_count = 0
    clean_segment_count = 0
    
    while trig_count < len(triggers) - num_cycles_for_template: # loop until end, stop before template will overlap with last trigger
        
        print('Trigger ' + str(trig_count) + ' of ' + str(len(triggers)))
        
        artefact_segment_start_time = triggers[trig_count] # start time of segment of data to be cleaned
        
        artefact_segment = data[artefact_segment_start_time:artefact_segment_start_time+length_artefact_segment] # segment of data to be cleaned
    
       # print('Segment range = ' + str(int(np.ptp(artefact_segment))))
    
        
        segment_range = np.ptp(artefact_segment)
        
 
        template_count = 0
        k = triggers[0] + 20 # start from the begining of the triggers plus 20 
    
    
        if segment_range > 1500: # only try to remove artefact if the range of the segment to be cleaned is above this threshold

            template_matrix = np.zeros([num_templates,length_artefact_segment]) # empty matrix to put clean (long) segments into

            # collect segments to make the artefact template
            while (template_count < num_templates) and (k < triggers[-1]-length_artefact_segment): # loop until enough templates or the end of the triggers
         
                temp_template = data[k:k+length_artefact_segment]
        
                if np.corrcoef(artefact_segment,temp_template)[0,1] > correlation_threshold:
       
                    ## check up to 20 data points ahead and behind for the best correlation
                    
                    temp_corr_scores = np.zeros([40,]) # keep track of correlation scores
                    
                    count = 0
                    for t in range(-20,20):
                        temp_template = data[k+t:k+t+length_artefact_segment]
                      #  plt.plot(temp_template,'c')
                        temp_corr_scores[count] = np.corrcoef(artefact_segment,temp_template)[0,1]
                        count += 1
                        
                    best_time_index = np.argmax(temp_corr_scores) - 20 # choose the segment with the best correlation
                    
                    best_template = data[k+best_time_index:k+best_time_index+length_artefact_segment]
                    
                    #don't include the template if the correlation is 1, because this is the segment we are trying to clean
                    if np.corrcoef(artefact_segment,best_template)[0,1] < 1: 
                        
                        template_matrix[template_count,:] = best_template
                        template_count += 1
                
                    best_template = best_template - best_template.mean()
                    
                   # plt.plot(best_template,'r')
                    
                    k = k + 500 # skip ahead to save time, becasue another closely matching template is unlikely to follow immediatly, e.g. walking artefact is typically > 1Hz
                    
                k += 2 # move forward, more than on data point is OK because the initial template fit only needs to be approximate, the code will scan back and forward for the best fit
    
        if template_count > 5: # only average to make the template if there are more than 5 segments of data contributing to the template
            # average all the segments to make the template 
            average_template = template_matrix[0:template_count,:].mean(axis=0)
               
            # subtract the template from the segment to be cleaned
            cleaned_artefact_segment = artefact_segment - average_template
            
        else: # if there are not enough template segments, just use the original segment of data
            cleaned_artefact_segment = artefact_segment
            
    
    
        ## put all the flicker segments from the cleaned segment into the clean segment matrix
        trigger_time = artefact_segment_start_time
        while trigger_time <= (artefact_segment_start_time + length_artefact_segment):
            
            segment = cleaned_artefact_segment[trigger_time - artefact_segment_start_time:trigger_time - artefact_segment_start_time+period]
            
            if (len(segment) == period) and (np.ptp(segment) < 500): # check segment is the correct length and is within range of 200
            
                clean_segment_matrix[clean_segment_count,:] = segment # put clean segment into matrix
                clean_segment_count += 1
                
            
            trig_count += 1
            trigger_time = triggers[trig_count] # move onto the next trigger
            
    
        print(str(clean_segment_count) + ' good segments')
    
    
    clean_SSVEP = clean_segment_matrix[0:clean_segment_count,:].mean(axis=0) # average segments to make the SSVEP
    
    clean_SSVEP = clean_SSVEP - clean_SSVEP.mean() # baseline correct
    
    return clean_SSVEP




### make SSVEP with motion artefact removal based on pairing segments

def make_SSVEP_artefact_removal_paired_segments(data, triggers, period):
    
    import numpy as np
    
    range_cutoff = 900
    
    
    all_segments_matrix = np.zeros([len(triggers), period]) # empty matrix to put segments into
    
    seg_count = 0 # keep track of the number of segments
    
    # loop through all triggers and put the corresponding segment of data into the matrix
    for trigger in triggers:
    
        # select a segment of data the lenght of the flicker period, starting from the trigger time 
        segment =  data[trigger:trigger+period] 
        
        all_segments_matrix[seg_count,:] = segment
        
        seg_count += 1
     
    segments_to_use = [] # list 
    segments_not_yet_used = list(range(0,len(triggers)))


   # for each segment, try and find a segment with the lowest difference, e.g. the range of the summed wave closest to 0
    for seg_count in range(0,len(triggers)):
        
        print('Segment ' + str(seg_count) + ' of ' + str(len(triggers)))
        
        if seg_count in segments_not_yet_used:
            
            segment = all_segments_matrix[seg_count,:]
            
            range_of_sum_scores = np.zeros([len(triggers),]) + 1000 # set all to arbitary high value first
            
            for seg_num in segments_not_yet_used:
                
                test_segment = all_segments_matrix[seg_num,:]
                
                range_of_sum_scores[seg_num] = np.ptp(segment + test_segment)

            print(min(range_of_sum_scores)) 
           
            if min(range_of_sum_scores) < range_cutoff:
                
                best_seg = np.argmin(range_of_sum_scores)
                
                segments_to_use.append(seg_count)
                segments_to_use.append(best_seg)
                
                if seg_count in segments_not_yet_used:
                    segments_not_yet_used.remove(seg_count)
                if best_seg in segments_not_yet_used:
                    segments_not_yet_used.remove(best_seg)
                    
                    
    print(' ')          
    print('Used ' + str(len(segments_to_use)) + ' of ' + str(len(triggers)) + ' triggers')          
    
    
    SSVEP = all_segments_matrix[segments_to_use,:].mean(axis=0) # average to make SSVEP
    
    SSVEP = SSVEP - SSVEP.mean() # baseline correct

    # true_SSVEP = all_segments_matrix.mean(axis=0) # average to make SSVEP
   
    # true_SSVEP = true_SSVEP - true_SSVEP.mean()
    
    # plt.plot(true_SSVEP)
    # plt.plot(SSVEP)
    
   
    return SSVEP

    
