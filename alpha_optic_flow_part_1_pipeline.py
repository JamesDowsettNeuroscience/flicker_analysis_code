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
#freq_bins = np.arange(72,125,5) # make frequency bins

#freq_bins = 1000/np.arange(8,14,0.2) # make frequency bins

freq_bins = 1000/np.arange(8,14,0.5) # make frequency bins

number_of_triggers_per_bin = 200



all_SSVEPs = np.zeros([num_subjects, 8, 3, 2, len(freq_bins), int(max(freq_bins))]) # subject, electrode, condition, block, freq bins, SSVEP data

all_SSVEP_amplitudes = np.zeros([num_subjects, 8, 3, 2, len(freq_bins)])

trials_with_no_triggers = []
trials_with_low_trigger_count = []

subject_numbers_with_low_trigger_counts = []

subject_count = 0

for subject in range(1,21):
    
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
                subject_numbers_with_low_trigger_counts.append(subject_count)
            else:
            
  
                ######## sort triggers by period into different frequency bins ##########
                
                
                ## give each trigger a true frequency value, interpolated from integer period values with sliding window
                diff_triggers = np.diff(good_triggers)     
    
                true_flicker_periods = np.zeros([len(diff_triggers),])
            
                # estimate the periods for the first 50, starting at 125, linear interpolate to the 50th value
                sliding_selection = good_triggers[0:100]
                diff_sliding_selection = np.diff(sliding_selection)
                flicker_periods = diff_sliding_selection[diff_sliding_selection < 128]
                period = flicker_periods.mean()
                
                true_flicker_periods[0:50] = np.linspace(125,period,50)
                
                # estimate the true flicker period with sliding window
                k = 50
                while k < len(good_triggers)-50:
                    
                    sliding_selection = good_triggers[k-50:k+50]
                    diff_sliding_selection = np.diff(sliding_selection)
                    flicker_periods = diff_sliding_selection[diff_sliding_selection < 128]
                 
                    period = flicker_periods.mean()
                    true_flicker_periods[k] = period
                
                    k += 1
         
                true_flicker_periods[k:] = np.linspace(period,114,49)
                
                
                
                
                
               # trigger_time_series = np.zeros([len(EOG_data),len(freq_bins)])
                
                for current_bin in range(0,len(freq_bins)):
                    
                    period = freq_bins[current_bin] # period of the current frequency bin
                    
                   # get the index of the triggers whose period is closest to current bin period
                    idx = np.argsort(np.abs(true_flicker_periods-period)) 
                    
                    closest_trigger_indices = idx[:number_of_triggers_per_bin].tolist()
                                 
                    triggers_for_this_bin = [good_triggers[i] for i in closest_trigger_indices]
                   
                   # triggers_for_this_bin = []
    
                   #  for trig_count in range(0,len(good_triggers)-1):
                        
       
                   #      if diff_triggers[trig_count] < 128:
                            
                   #          # get the frequency bin with the period closest to the period of the flicker trigger
                   #          freq_bin = np.argmin(np.abs(freq_bins - diff_triggers[trig_count]))
                
                   #          if freq_bin == current_bin:
                   #              triggers_for_this_bin.append(good_triggers[trig_count])
                   #             # trigger_time_series[good_triggers[trig_count],current_bin] = 1
                                
                    num_triggers_for_bin = len(triggers_for_this_bin)
                    
                    print(str(num_triggers_for_bin) + ' segments ' + str(period) + ' ms = ' + str(np.round(sample_rate/freq_bins[current_bin],2)) + ' Hz')
                    
                  #  numbers_of_triggers_per_bin[current_bin] = num_triggers_for_bin
                    
                    ## save the triggers
                    triggers_for_this_bin_array = np.asarray(triggers_for_this_bin)
                    np.save(path + trial_name + '_bin_' + str(current_bin), triggers_for_this_bin_array)
                    
                 #   plt.suptitle(str(num_triggers_for_bin) + ' segments ' + str(period) + ' ms = ' + str(np.round(sample_rate/freq_bins[current_bin],2)) + ' Hz')
                    
            
                    
            
                    for electrode in range(0,8):
                        
                        # plt.subplot(2,3,electrode+1)
                        # plt.title(electrode_names[electrode])
                        
                        data = all_data[electrode,:]
                        
                        data = data - (REF_2_data/2) # re-reference
                        
                        SSVEP = functions.make_SSVEPs(data, triggers_for_this_bin, int(period))
                        
                        # subject, electrode, condition, block, freq bins, SSVEP data
                        all_SSVEPs[subject_count,electrode, condition-1, block-1, current_bin, 0:int(period)] = SSVEP
            
                        all_SSVEP_amplitudes[subject_count,electrode, condition-1, block-1, current_bin] = np.ptp(SSVEP)
                        
                        #plt.plot(data, label = electrode_names[electrode])
                      #  plt.plot(SSVEP, label = block)
                    
                    #plt.legend()
                    
                    
                # if min(numbers_of_triggers_per_bin) < 50:
                #     trials_with_low_trigger_count.append(trial_name)
                #     subject_numbers_with_low_trigger_counts.append(subject_count)
                    
                
    subject_count += 1
       
         
 # remove subjects with low trigger counts. These subjects can be ignored             
# remove duplicates             
subject_numbers_with_low_trigger_counts = list(dict.fromkeys(subject_numbers_with_low_trigger_counts)) 

subjects_to_use = list(range(20))    

for bad_subject in subject_numbers_with_low_trigger_counts:
    subjects_to_use.remove(bad_subject)








##########   get individual alpha frequencies   ###############################


electrode = 0

length = 5 # length of FFT in seconds


plt.figure()
plt.suptitle('Electrode: ' + electrode_names[electrode] + '  Freq resolution = ' + str(1/length) )


freq_axis = np.arange(0,sample_rate,1/length)

condition_colours = ('b', 'r', 'g')

subject_count = 0

alpha_peaks = np.zeros([num_subjects, 3])
IAFs = np.zeros([num_subjects, 3])

all_induced_FFTs = np.zeros([num_subjects, 3, length*sample_rate])

subject_count = 0

for subject in range(1,21):
    
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
        
        all_induced_FFTs[subject_count, condition-1,:] = induced_fft
        
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
                  
                 







############   Make simulated data ####################

## add simulated SSVEPs to resting data 

for subject in range(1,21):
    
    print('\n Subject ' + str(subject) + '\n')
    
    for condition in(1,2,3):
        
        print('Condition ' + str(condition))
        

        # load the resting data from block 3,
        trial_name = 'subject_' + str(subject) + '_block_3_condition_' + str(condition) 
        
        all_data = np.load(path + trial_name + '_all_data.npy')

        #
        all_simulated_data = np.copy(all_data)

        # load the triggers from block 1
        trial_name = 'subject_' + str(subject) + '_block_1_condition_' + str(condition) 
        
        triggers = np.load(path + trial_name + '_all_triggers.npy')

        for electrode in range(0,8):
        
            simulated_data = all_simulated_data[electrode,:]
    
            # use the mean amplitude of SSVEPs across all conditions and frequency bins
            amplitudes_all_bins = all_SSVEP_amplitudes[subject-1, electrode,:,0,:]
            
            simulated_SSVEP_amplitude = amplitudes_all_bins.mean()
    
            for trigger_count in range(0,len(triggers)-1):
                
                time_to_next_trigger = triggers[trigger_count+1] - triggers[trigger_count]
                
                if time_to_next_trigger < 128 and (triggers[trigger_count+1] < len(simulated_data)):
                    
                    t = np.arange(0,time_to_next_trigger)
                    
                    ## control simulated SSVEP - no difference between conditions
                    simulated_SSVEP = simulated_SSVEP_amplitude * np.sin(2 * np.pi * 1/sample_rate * (sample_rate/time_to_next_trigger) * t)
                
                    ## simulated SSVEPs with an effect of condition
                  #  simulated_SSVEP = simulated_SSVEP_amplitude * np.sin(2 * np.pi * 1/sample_rate * (sample_rate/time_to_next_trigger) * (t+(condition*10)))
                
                    simulated_data[triggers[trigger_count]:triggers[trigger_count]+time_to_next_trigger] = simulated_SSVEP + simulated_data[triggers[trigger_count]:triggers[trigger_count]+time_to_next_trigger]
    
            all_simulated_data[electrode,:] = simulated_data    

        # save simulated_data
        trial_name = 'subject_' + str(subject) + '_simulated_condition_' + str(condition) + '_all_data'
        
        np.save(path + trial_name, all_simulated_data)






##### compare SSVEP amplitudes with resting FFT  ########


flicker_frequencies_per_bin = np.round(1000/freq_bins,1)

# for electrode in range(0,6):
    
#     plt.subplot(3,2,electrode+1)
#     plt.title(electrode_names[electrode])
        
        
#     for condition in range(0,3):
        
#         # subject, electrode, condition, block, freq bins
#         block_1_SSVEP_amplitudes = all_SSVEP_amplitudes[subjects_to_use,electrode,condition, 0,:]
#         block_2_SSVEP_amplitudes = all_SSVEP_amplitudes[subjects_to_use,electrode,condition, 1,:]
    
#         average_across_participants_block_1 = np.nanmean(block_1_SSVEP_amplitudes, axis=0)
#         average_across_participants_block_2 = np.nanmean(block_2_SSVEP_amplitudes, axis=0)
                 
#         average_across_blocks = (average_across_participants_block_1 + average_across_participants_block_2) / 2
        
                 
#         plt.plot(flicker_frequencies_per_bin,average_across_blocks*350, color = condition_colours[condition])
                
#         all_subjects_FFTs = all_induced_FFTs[subjects_to_use, condition,:]
#         average_FFT = all_subjects_FFTs.mean(axis=0)
                
#         plt.plot(freq_axis,average_FFT, color = condition_colours[condition])
    
#         plt.xlim([4, 16])
#         plt.ylim([0, 0.0025])






# ###### check block 1 and 2 correlations ##############

# block_1_and_2_correlations = np.zeros([len(subjects_to_use),6,3,len(freq_bins)])

# subject_count = 0
# for subject in subjects_to_use:
    
#     for electrode in range(0,6):
    
#         for condition in range(0,3):
    
#             for current_bin in range(0,len(freq_bins)):
                
#                 period = freq_bins[current_bin]

#                 # subject, electrode, condition, block, freq bins, SSVEP data
#                 block_1_SSVEP = all_SSVEPs[subject, electrode, condition, 0, current_bin, 0:period]
#                 block_2_SSVEP = all_SSVEPs[subject, electrode, condition, 1, current_bin, 0:period]
                
#                 block_1_and_2_correlations[subject_count, electrode, condition, current_bin] = np.corrcoef(block_1_SSVEP,block_2_SSVEP)[0,1]
    
#     subject_count += 1
    
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
    





# ######### plot P3/P4 ratios ############


# P3_P4_ratios = np.zeros([len(subjects_to_use),3, len(freq_bins)])

# plot_colors = ('b', 'r', 'g')

# subject_count = 0

# for subject in subjects_to_use:
    
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
            
#             P3_P4_ratios[subject_count, condition, current_bin] = P3_P4_ratio
            
#             # print(P3_P4_ratio)
#     subject_count += 1
            
            
# average_P3_P4_ratios = P3_P4_ratios[:,:,:].mean(axis=0)



# plt.figure()
# plt.title('Average P3 / P4 ratios')

# for condition in range(0,3):
    
#     plt.plot(1000/freq_bins,average_P3_P4_ratios[condition,:], label = condition_names[condition], color = plot_colors[condition])

#     # for current_bin in range(0,len(freq_bins)):
        
#       #  plt.scatter(np.ones([subject_count,])*(1000/freq_bins[current_bin])+(condition*0.05),P3_P4_ratios[:,condition,current_bin], s=1, c=plot_colors[condition])


# plt.legend()        

# plt.plot(1000/freq_bins,np.ones(len(freq_bins)), 'k--')

# plt.xlabel('Flicker Frequency (Hz)')
# plt.ylabel('P3/P4 Ratio')

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
            
            
            



################ Decoding #####################

plt.figure()

for electrode in range(0,8):
    
    decoding_accuracy = np.zeros([num_subjects,8,3, len(freq_bins)])
    
    #decoding_accuracy = np.zeros([num_subjects,6, len(freq_bins)])
      

    for block in (1,2,3):
        
        subject_count = 0
        
        for subject_number in subjects_to_use:
            
            subject = subject_number+1
            
            print('\n Subject ' + str(subject) + '  ' + electrode_names[electrode] + '  Block: ' + str(block) + '\n')
            
            if block == 1 or block == 2:
            # load data for condition 1
                trial_name = 'subject_' + str(subject) + '_block_' + str(block) + '_condition_1' 
                all_data = np.load(path + trial_name + '_all_data.npy')
                data_1 = all_data[electrode,:]
                REF_2_data = all_data[6,:] # get the data for the second reference, to re-reference (left and right ears)
                data_1 = data_1 - (REF_2_data/2) # re-reference
                
                # load data for condition 2
                trial_name = 'subject_' + str(subject) + '_block_' + str(block) + '_condition_2' 
                all_data = np.load(path + trial_name + '_all_data.npy')
                data_2 = all_data[electrode,:]
                REF_2_data = all_data[6,:] # get the data for the second reference, to re-reference (left and right ears)
                data_2 = data_2 - (REF_2_data/2) # re-reference
                
                # load data for condition 3
                trial_name = 'subject_' + str(subject) + '_block_' + str(block) + '_condition_3' 
                all_data = np.load(path + trial_name + '_all_data.npy')
                data_3 = all_data[electrode,:]
                REF_2_data = all_data[6,:] # get the data for the second reference, to re-reference (left and right ears)
                data_3 = data_3 - (REF_2_data/2) # re-reference
                
            else: # simulated data
            
                # load data for condition 1
                trial_name = 'subject_' + str(subject) + '_simulated_condition_1_all_data'
                all_data = np.load(path + trial_name + '.npy')
                data_1 = all_data[electrode,:]
                REF_2_data = all_data[6,:] # get the data for the second reference, to re-reference (left and right ears)
                data_1 = data_1 - (REF_2_data/2) # re-reference
                
                # load data for condition 2
                trial_name = 'subject_' + str(subject) + '_simulated_condition_2_all_data'
                all_data = np.load(path + trial_name + '.npy')
                data_2 = all_data[electrode,:]
                REF_2_data = all_data[6,:] # get the data for the second reference, to re-reference (left and right ears)
                data_2 = data_2 - (REF_2_data/2) # re-reference
                
                # load data for condition 3
                trial_name = 'subject_' + str(subject) + '_simulated_condition_3_all_data'
                all_data = np.load(path + trial_name + '.npy')
                data_3 = all_data[electrode,:]
                REF_2_data = all_data[6,:] # get the data for the second reference, to re-reference (left and right ears)
                data_3 = data_3 - (REF_2_data/2) # re-reference
    
    
            # # load and concatenate data for condition 1
            # trial_name = 'subject_' + str(subject) + '_block_1_condition_1' 
            # all_data = np.load(path + trial_name + '_all_data.npy')
            # data_1_block_1 = all_data[electrode,:]
            
            # trial_name = 'subject_' + str(subject) + '_block_2_condition_1' 
            # all_data = np.load(path + trial_name + '_all_data.npy')
            # data_1_block_2 = all_data[electrode,:]           
            
            # data_1 = np.concatenate((data_1_block_1, data_1_block_2))
           
            # # load and concatenate data for condition 2
            # trial_name = 'subject_' + str(subject) + '_block_1_condition_2' 
            # all_data = np.load(path + trial_name + '_all_data.npy')
            # data_2_block_1 = all_data[electrode,:]
            
            # trial_name = 'subject_' + str(subject) + '_block_2_condition_2' 
            # all_data = np.load(path + trial_name + '_all_data.npy')
            # data_2_block_2 = all_data[electrode,:]           
            
            # data_2 = np.concatenate((data_2_block_1, data_2_block_2))
                
            # # load and concatenate data for condition 3
            # trial_name = 'subject_' + str(subject) + '_block_1_condition_3' 
            # all_data = np.load(path + trial_name + '_all_data.npy')
            # data_3_block_1 = all_data[electrode,:]
            
            # trial_name = 'subject_' + str(subject) + '_block_2_condition_3' 
            # all_data = np.load(path + trial_name + '_all_data.npy')
            # data_3_block_2 = all_data[electrode,:]           
            
            # data_3 = np.concatenate((data_1_block_1, data_1_block_2))
                   
           
            
            for current_bin in range(0,len(freq_bins)):
                
                if block == 1 or block == 3: # use triggers from block 1 for block 1 and simulated data
                    
                    # load  triggers for condition 1
                    trial_name_1 = 'subject_' + str(subject) + '_block_1_condition_1' 
                    triggers_1 = np.load(path + trial_name_1 + '_bin_' + str(current_bin) + '.npy')
                    
                    # load  triggers for condition 2
                    trial_name_2 = 'subject_' + str(subject) + '_block_1_condition_2' 
                    triggers_2 = np.load(path + trial_name_2 + '_bin_' + str(current_bin) + '.npy')
                    
                    # load  triggers for condition 3
                    trial_name_3 = 'subject_' + str(subject) + '_block_1_condition_3' 
                    triggers_3 = np.load(path + trial_name_3 + '_bin_' + str(current_bin) + '.npy')
                    
                elif block == 2:
                    
                    # load  triggers for condition 1
                    trial_name_1 = 'subject_' + str(subject) + '_block_2_condition_1' 
                    triggers_1 = np.load(path + trial_name_1 + '_bin_' + str(current_bin) + '.npy')
                    
                    # load  triggers for condition 2
                    trial_name_2 = 'subject_' + str(subject) + '_block_2_condition_2' 
                    triggers_2 = np.load(path + trial_name_2 + '_bin_' + str(current_bin) + '.npy')
                    
                    # load  triggers for condition 3
                    trial_name_3 = 'subject_' + str(subject) + '_block_2_condition_3' 
                    triggers_3 = np.load(path + trial_name_3 + '_bin_' + str(current_bin) + '.npy')
                    
                    
    
    
    
                # # load and concatenate data for condition 1
                # trial_name_1_block_1 = 'subject_' + str(subject) + '_block_1_condition_1' 
                # trial_name_1_block_2 = 'subject_' + str(subject) + '_block_2_condition_1' 
                # triggers_1_block_1 = np.load(path + trial_name_1_block_1 + '_bin_' + str(current_bin) + '.npy')
                # triggers_1_block_2 = np.load(path + trial_name_1_block_2 + '_bin_' + str(current_bin) + '.npy')
            
                # triggers_1 = np.concatenate((triggers_1_block_1,triggers_1_block_2+len(data_1_block_2)))
            
                # # load and concatenate data for condition 2
                # trial_name_2_block_1 = 'subject_' + str(subject) + '_block_1_condition_2' 
                # trial_name_2_block_2 = 'subject_' + str(subject) + '_block_2_condition_2' 
                # triggers_2_block_1 = np.load(path + trial_name_2_block_1 + '_bin_' + str(current_bin) + '.npy')
                # triggers_2_block_2 = np.load(path + trial_name_2_block_2 + '_bin_' + str(current_bin) + '.npy')
            
                # triggers_2 = np.concatenate((triggers_2_block_1,triggers_2_block_2+len(data_2_block_2)))
            
    
                # # load and concatenate data for condition 3
                # trial_name_3_block_1 = 'subject_' + str(subject) + '_block_1_condition_3' 
                # trial_name_3_block_2 = 'subject_' + str(subject) + '_block_2_condition_3' 
                # triggers_3_block_1 = np.load(path + trial_name_3_block_1 + '_bin_' + str(current_bin) + '.npy')
                # triggers_3_block_2 = np.load(path + trial_name_3_block_2 + '_bin_' + str(current_bin) + '.npy')
            
                # triggers_3 = np.concatenate((triggers_3_block_1,triggers_3_block_2+len(data_3_block_2)))
    
    
                ## decode with universal decoder function
                
                # first put all data  into one matrix
                max_data_length = max(len(data_1), len(data_2), len(data_3))
                
                data_all_conditions = np.zeros([3,max_data_length])
                data_all_conditions[0,0:len(data_1)] = data_1
                data_all_conditions[1,0:len(data_2)] = data_2
                data_all_conditions[2,0:len(data_3)] = data_3
                  
                # put all triggers into one matrix
                triggers_all_conditions = np.zeros([3,number_of_triggers_per_bin])
                triggers_all_conditions = triggers_all_conditions.astype(int)
                triggers_all_conditions[0,:] = triggers_1
                triggers_all_conditions[1,:] = triggers_2
                triggers_all_conditions[2,:] = triggers_3
               
                num_triggers = number_of_triggers_per_bin
                num_loops = 10
                
                average_percent_correct = functions.decode_correlations(data_all_conditions, triggers_all_conditions, 3, num_triggers, int(period), num_loops)
                
                # make SSVEPs
                SSVEP_1 = functions.make_SSVEPs(data_1, triggers_1, int(freq_bins[current_bin]))
                SSVEP_2 = functions.make_SSVEPs(data_2, triggers_2, int(freq_bins[current_bin]))
                SSVEP_3 = functions.make_SSVEPs(data_3, triggers_3, int(freq_bins[current_bin]))
                
                period = freq_bins[current_bin]
                num_loops = 10
                
                average_percent_correct = functions.decode_correlation_3way(data_1, data_2, data_3, triggers_1, triggers_2, triggers_3, int(period), num_loops)
                
                decoding_accuracy[subject_count,electrode,block-1,current_bin] = average_percent_correct
                #decoding_accuracy[subject_count,electrode,current_bin] = average_percent_correct
                
                print(str(1000/freq_bins[current_bin]) + 'Hz  Average percent correct = ' + str(average_percent_correct))
                
                    
            subject_count += 1            
                
            

            
# for subject in range(0,20):
    
#     re_scale = alpha_peaks[subject,2] / 100 # re-scale the plot so 100% is the same as condition 3 peak
    
#     decoding_scores_for_block_1 = decoding_accuracy[subject,electrode,0,:] * re_scale
#     decoding_scores_for_block_2 = decoding_accuracy[subject,electrode,1,:] * re_scale
    
#     average_decoding_scores = (decoding_scores_for_block_1 + decoding_scores_for_block_2) /2
    
#     plt.subplot(4,5,subject+1)
#     plt.title('Subject ' + str(subject+1))       
    
#     # plt.plot(flicker_frequencies_per_bin, decoding_scores_for_block_1)
#     # plt.plot(flicker_frequencies_per_bin, decoding_scores_for_block_2)
    
#     plt.plot(flicker_frequencies_per_bin, average_decoding_scores,'k')

        
## plot grand average
    
    decoding_scores_for_block_1 = decoding_accuracy[:,electrode,0,:]
    average_decoding_scores_block_1 = decoding_scores_for_block_1.mean(axis=0)
    
    plt.subplot(1,3,1)
    plt.plot(flicker_frequencies_per_bin, average_decoding_scores_block_1)
    plt.ylim([0, 100]) 
    plt.axhline(y = 33.33, color = 'k', linestyle = '--')
    
    decoding_scores_for_block_2 = decoding_accuracy[:,electrode,1,:]
    average_decoding_scores_block_2 = decoding_scores_for_block_2.mean(axis=0)
    
    plt.subplot(1,3,2)
    plt.plot(flicker_frequencies_per_bin, average_decoding_scores_block_2)
    plt.ylim([0, 100]) 
    plt.axhline(y = 33.33, color = 'k', linestyle = '--')
    
    decoding_scores_for_block_3 = decoding_accuracy[:,electrode,2,:]
    average_decoding_scores_block_3 = decoding_scores_for_block_3.mean(axis=0)
    
    plt.subplot(1,3,3)
    plt.plot(flicker_frequencies_per_bin, average_decoding_scores_block_3, label = electrode_names[electrode])
    plt.ylim([0, 100]) 
    plt.axhline(y = 33.33, color = 'k', linestyle = '--')
        
        
    
    # grand_average_decoding_scores = (average_decoding_scores_block_1 + average_decoding_scores_block_2) /2

    # plt.subplot(1,3,3)
    # plt.plot(flicker_frequencies_per_bin, grand_average_decoding_scores, label = electrode_names[electrode])
    
  
    # average_decoding_scores = decoding_accuracy[:,electrode,:].mean(axis=0)
    
    # plt.plot(flicker_frequencies_per_bin, average_decoding_scores, label = electrode_names[electrode])
    

plt.legend()
plt.suptitle('Number segments per bin = ' + str(number_of_triggers_per_bin))



