#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:48:54 2023

@author: James Dowsett
"""

### alpha optic flow part 1 analysis pipeline ######


from flicker_analysis_package import functions

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

import random

### information about the experiment:

path = '/media/james/Expansion/alpha_optic_flow/experiment_1_data/'

condition_names = ('Eyes Track, head fixed', 'Head track, Eyes fixed', 'Both head and eyes track', 'Eyes Track, head fixed - control', 'Head track, Eyes fixed - control', 'Both head and eyes track - control')

electrode_names = ('O1', 'O2', 'P3', 'P4', 'P7', 'P8', 'REF2', 'EOG')

condition_names = ('Optic flow', 'Random', 'Static')

sample_rate = 1000

num_subjects = 20

length = 1 # length of FFT in seconds


## frequency bins to sort triggers into
freq_bins = np.arange(72,125,5) # make frequency bins

all_SSVEPs = np.zeros([num_subjects, 6, 3, 2, len(freq_bins), max(freq_bins)]) # subject, electrode, condition, block, freq bins, SSVEP data

trials_with_no_triggers = []
trials_with_low_trigger_count = []


subjects_to_use = (1,2,3,5,6,7,8,9,10,11,12,13,15,16,17,19,20)

subject_count = 0

for subject in subjects_to_use:
    
    print('\n Subject ' + str(subject) + '\n')
    
    for condition in(1,2,3):
        
        print('\n Condition ' + str(condition) + '\n')
        
        for block in(1,2):
            
            print('\n Block ' + str(block) + '\n')
            
        
            trial_name = 'subject_' + str(subject) + '_block_' + str(block) + '_condition_' + str(condition) 
            
            all_data = np.load(path + trial_name + '_all_data.npy')
            
            triggers = np.load(path + trial_name + '_all_triggers.npy')
            
            all_acc_data = np.load(path + trial_name + '_all_acc_data.npy')
            
            REF_2_data = all_data[6,:] # get the data for the second reference, to re-reference (left and right ears)
            
                        ## make time series to check triggers
            # trigger_time_series = np.zeros([len(EOG_data),])
            # for trigger in triggers:
            #     trigger_time_series[trigger] = 0.0001
            
            # plt.plot(trigger_time_series)
            # plt.plot(EOG_data)
            
            
            
            ## check for eye blinks
            EOG_data = all_data[7,:]
            EOG_data = EOG_data - EOG_data.mean() # baseline correct
            
            EOG_data = functions.low_pass_filter(10, sample_rate, EOG_data)
            

            good_triggers = []
            
            for trigger in triggers:
                if np.ptp(EOG_data[trigger-100:trigger+100]) < 0.00005:
                    good_triggers.append(trigger)
            
            ## make time series of good triggers to check eye blink removal worked
            # good_trigger_time_series = np.zeros([len(EOG_data),])
            # for trigger in good_triggers:
            #     good_trigger_time_series[trigger] = 0.0001
           # plt.plot(good_trigger_time_series)
            
            print(str(len(good_triggers)) + '  good triggers\n')   
            if len(good_triggers) == 0:
                print('Error: no triggers')
                trials_with_no_triggers.append(trial_name)
            
            
            
            
            ######## sort triggers by period into different frequency bins ##########

            diff_triggers = np.diff(good_triggers)     
            
            numbers_of_triggers_per_bin = np.zeros([len(freq_bins),])
            
           # trigger_time_series = np.zeros([len(EOG_data),len(freq_bins)])
            
            for current_bin in range(0,len(freq_bins)):
                
            #    plt.figure(current_bin)
                
                period = freq_bins[current_bin]
               
                triggers_for_this_bin = []
                
                for trig_count in range(0,len(good_triggers)-1):
                    
   
                    if diff_triggers[trig_count] < 128:
                        
                        # get the frequency bin with the period closest to the period of the flicker trigger
                        freq_bin = np.argmin(np.abs(freq_bins - diff_triggers[trig_count]))
            
                        if freq_bin == current_bin:
                            triggers_for_this_bin.append(good_triggers[trig_count])
                           # trigger_time_series[good_triggers[trig_count],current_bin] = 1
                            
                num_triggers_for_bin = len(triggers_for_this_bin)
                print(str(num_triggers_for_bin) + ' segments ' + str(period) + ' ms = ' + str(np.round(sample_rate/freq_bins[current_bin],2)) + ' Hz')
                
                numbers_of_triggers_per_bin[current_bin] = num_triggers_for_bin
                
                
                
             #   plt.suptitle(str(num_triggers_for_bin) + ' segments ' + str(period) + ' ms = ' + str(np.round(sample_rate/freq_bins[current_bin],2)) + ' Hz')
                
        
                
        
                for electrode in range(0,6):
                    
                    # plt.subplot(2,3,electrode+1)
                    # plt.title(electrode_names[electrode])
                    
                    data = all_data[electrode,:]
                    
                    data = data - (REF_2_data/2) # re-reference
                    
                    SSVEP = functions.make_SSVEPs(data, triggers_for_this_bin, period)
                    
                    # subject, electrode, condition, block, freq bins, SSVEP data
                    all_SSVEPs[subject_count,electrode, condition-1, block-1, current_bin, 0:period] = SSVEP
        
                    
                    #plt.plot(data, label = electrode_names[electrode])
                  #  plt.plot(SSVEP, label = block)
                
                #plt.legend()
                
                
            if min(numbers_of_triggers_per_bin) < 50:
                trials_with_low_trigger_count.append(trial_name)
                
                
                
    subject_count += 1
                
                
              
 



##########   get individual alpha frequencies   ###############################

subjects_to_use = np.arange(1,21)

electrode = 0

length = 5 # length of FFT in seconds


plt.figure()
plt.suptitle('Electrode: ' + electrode_names[electrode] + '  Freq resolution = ' + str(1/length) )


freq_axis = np.arange(0,sample_rate,1/length)

condition_colours = ('b', 'r', 'g')

subject_count = 0

alpha_peaks = np.zeros([len(subjects_to_use), 3])
IAFs = np.zeros([len(subjects_to_use), 3])


subject_count = 0

for subject in subjects_to_use:
    
    print('\n Subject ' + str(subject) + '\n')
      
    plt.subplot(4,5,subject)
   
    
    
    for condition in(1,2,3):
        
        print('Condition ' + str(condition))
        
        block = 3

        trial_name = 'subject_' + str(subject) + '_block_' + str(block) + '_condition_' + str(condition) 
        
        all_data = np.load(path + trial_name + '_all_data.npy')
        
        REF_2_data = all_data[6,:] # get the data for the second reference, to re-reference (left and right ears)

        #for electrode in range(0,6):
            
        data = all_data[electrode,:]
            
        data = data - (REF_2_data/2) # re-reference
            
        induced_fft = functions.induced_fft(data, length, sample_rate)
        
        plt.plot(freq_axis, induced_fft, color = condition_colours[condition-1])
        
        alpha_range = induced_fft[8*length:14*length]
        
        alpha_peak = max(alpha_range)
        alpha_peaks[subject_count, condition-1] = alpha_peak
        
        IAF = freq_axis[8*length + np.argmax(alpha_range)]
        IAFs[subject_count, condition-1] = IAF
        
        plt.plot(IAF, alpha_peak, '*', color = condition_colours[condition-1])
          
    plt.xlim([4, 16])
    plt.ylim([0, max(alpha_peaks[subject_count,:]) *1.1 ])
    
    plt.title('Subject: ' + str(subject) + ' ' + str(IAFs[subject_count, :]))
                  
    subject_count += 1
                  
                    
                  
             
                
             
                
             
                

###### check block 1 and 2 correlations ##############

# block_1_and_2_correlations = np.zeros([subject_count,6,3,len(freq_bins)])

# for subject in range(0,subject_count):
    
#     for electrode in range(0,6):
    
#         for condition in range(0,3):
    
#             for current_bin in range(0,len(freq_bins)):
                
#                 period = freq_bins[current_bin]

#                 # subject, electrode, condition, block, freq bins, SSVEP data
#                 block_1_SSVEP = all_SSVEPs[subject, electrode, condition, 0, current_bin, 0:period]
#                 block_2_SSVEP = all_SSVEPs[subject, electrode, condition, 1, current_bin, 0:period]
                
#                 block_1_and_2_correlations[subject, electrode, condition, current_bin] = np.corrcoef(block_1_SSVEP,block_2_SSVEP)[0,1]
    
    
# # plot block 1 and 2 correlations
# plt.figure()
# plt.suptitle('Average Block 1 : Block 2 correlations')
# for electrode in range(0,6):
#     plt.subplot(3,2,electrode+1)
#     plt.title(electrode_names[electrode])
#     plt.ylim([0, 1])
#     for condition in range(0,3):
        
#         correlations_all_subjects = block_1_and_2_correlations[:, electrode, condition, :]
    
#         average_correlations = correlations_all_subjects.mean(axis=0)
    
#         plt.plot(1000/freq_bins, average_correlations, label = condition_names[condition])
        
#         # for current_bin in range(0,len(freq_bins)):
            
#         #     plt.scatter(np.ones([subject_count,])*(1000/freq_bins[current_bin])+(condition*0.1),correlations_all_subjects[:,current_bin])
    
#     plt.legend()
    





######### plot P3/P4 ratios ############

# P3_P4_ratios = np.zeros([subject_count,3, len(freq_bins)])

# plot_colors = ('b', 'r', 'g')

# for subject in range(0,subject_count):
    
#     for condition in range(0,3):

#       #  plt.figure()
#         for current_bin in range(0,len(freq_bins)):

#             period = freq_bins[current_bin]
            
#             # subject, electrode, condition, block, freq bins, SSVEP data
            
#             # SSVEP_P3 = all_SSVEPs[subject, 2, condition,block, current_bin, 0:period]
#             # SSVEP_P4 = all_SSVEPs[subject, 3, condition,block, current_bin, 0:period]
            
#             block_1_SSVEP_P3 = all_SSVEPs[subject, 2, condition,0, current_bin, 0:period]
#             block_1_SSVEP_P4 = all_SSVEPs[subject, 3, condition,0, current_bin, 0:period]
            
#             block_2_SSVEP_P3 = all_SSVEPs[subject, 2, condition,1, current_bin, 0:period]
#             block_2_SSVEP_P4 = all_SSVEPs[subject, 3, condition,1, current_bin, 0:period]

#             SSVEP_P3 = (block_1_SSVEP_P3 + block_2_SSVEP_P3)/2
#             SSVEP_P4 = (block_1_SSVEP_P4 + block_2_SSVEP_P4)/2
            
#             # plt.subplot(3,4,current_bin+1)
#             # plt.title(freq_bins[current_bin])      
#             # plt.plot(average_SSVEP_P3)
#             # plt.plot(average_SSVEP_P4)
           
#             P3_P4_ratio = np.ptp(SSVEP_P3) / np.ptp(SSVEP_P4)
            
#             P3_P4_ratios[subject, condition, current_bin] = P3_P4_ratio
            
#            # print(P3_P4_ratio)
            
            
# average_P3_P4_ratios = P3_P4_ratios.mean(axis=0)



# plt.figure()
# plt.title('Average P3 / P4 ratios')

# for condition in range(0,3):
    
#     plt.plot(1000/freq_bins,average_P3_P4_ratios[condition,:], label = condition_names[condition], color = plot_colors[condition])

#    # for current_bin in range(0,len(freq_bins)):
        
#       #  plt.scatter(np.ones([subject_count,])*(1000/freq_bins[current_bin])+(condition*0.05),P3_P4_ratios[:,condition,current_bin], s=1, c=plot_colors[condition])


# plt.legend()        

# plt.plot(1000/freq_bins,np.ones(len(freq_bins)), 'k--')


# print('\nFrequency bins = ')
# print(1000/freq_bins)








#### plot accelorometer data

# length = 5 # length of FFT in seconds

# axis = 2

# freq_axis = np.arange(0,sample_rate,1/length)

# plt.figure()

# plt.suptitle('Freq resolution = ' + str(1/length) + '  Axis = ' + str(axis))

# condition_colours = ('b', 'r', 'g')

# for subject in range(1,21):
    
#     print('\n Subject ' + str(subject) + '\n')
    
#     plt.subplot(4,5,subject)
   
#     plt.title('Subject: ' + str(subject))
    
#     for condition in(1,2,3):
        
#         print('Condition ' + str(condition))
        
#         for block in(1,2):
            
#             print('Block ' + str(block))
            
#             trial_name = 'subject_' + str(subject) + '_block_' + str(block) + '_condition_' + str(condition) 

#             all_acc_data = np.load(path + trial_name + '_all_acc_data.npy')

#             #for axis in range(0,3):
            
                
#             axis_data = all_acc_data[axis,:]

#             fft_acc = functions.induced_fft(axis_data, length, sample_rate)


#             plt.plot(freq_axis,fft_acc, color = condition_colours[condition-1])
        
#             plt.xlim([2, 20])
#             plt.ylim([0, 2000])
            
            
