#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 14:19:31 2022

@author: James Dowsett
"""


#### Code to check the loss of quality in the final SSVEP, as a result of subtracting

import numpy as np
import matplotlib.pyplot as plt

import random

from flicker_analysis_package import functions

trial_time = 300 # total time in seconds

period = 111

time_vector = np.arange(0,period)

flicker_frequency = 9
sample_rate = 1000

phase_shift = 0

SSVEP_amplitude = 1


# one_cycle_SSVEP = SSVEP_amplitude * np.sin(2 * np.pi * flicker_frequency/sample_rate * (time_vector-phase_shift)) 

one_cycle_SSVEP = np.load('example_9Hz_SSVEP.npy')  
    
# repeat the single cycle at the correct frequency

number_of_flickers = int((trial_time*sample_rate)/period) # number of times the simulated flicker will repeat in 100 seconds

# empty array to put the simulated SSVEP into
simulated_SSVEP_data = np.zeros([trial_time * sample_rate])

# use tile to repeat the basic SSVEP
simulated_SSVEP_data[0:number_of_flickers*period] = np.tile(one_cycle_SSVEP,number_of_flickers )



triggers = np.arange(0, len(simulated_SSVEP_data)-period, period) # make triggers, stop one period length before the end


## load one segment of typical walking artefact
artefact_segment = np.load('example_artefact_segment.npy')

artefact_segment = artefact_segment - artefact_segment.mean() # baseline correct

number_of_walking_cycles =  int(len(simulated_SSVEP_data)/len(artefact_segment))  # int number of times the walking artefact will fit into the data
walking_artefact_data = np.zeros([trial_time * sample_rate])
walking_artefact_data[0:number_of_walking_cycles*len(artefact_segment)] = np.tile(artefact_segment,number_of_walking_cycles)


## combine all data

data = simulated_SSVEP_data + walking_artefact_data

# plot raw data

trigger_time_series = np.zeros([len(simulated_SSVEP_data)],)
for trigger in triggers:
    trigger_time_series[trigger] = 1


plt.figure()

# plt.plot(trigger_time_series)
    
# plt.plot(simulated_SSVEP_data)

# plt.plot(walking_artefact_data)

# plt.plot(data)


## make SSVEP

SSVEP = functions.make_SSVEPs(data,triggers,period)


plt.plot(one_cycle_SSVEP)

plt.plot(SSVEP)








#############  template subtraction for individual segments ######################


clean_segments_matrix = np.zeros([len(triggers),period])

clean_seg_count = 0

max_num_template_to_use = 20

range_cutoff = 300

time_range_to_search = 20000

seg_count = 0

for trigger in triggers:
    
    print('Segment ' + str(seg_count) + ' of ' + str(len(triggers)))
    
    segment_to_clean = data[trigger:trigger+period] 

    k = trigger + period
    
    template_matrix = np.zeros([max_num_template_to_use,period])
    template_count = 0
    
    while (k < trigger+time_range_to_search) and (template_count < max_num_template_to_use) and (k < triggers[-1]-time_range_to_search):
        
        temp_template = data[k:k+period]
        
        if np.ptp(segment_to_clean - temp_template) < range_cutoff:
            
            template_matrix[template_count,:] = temp_template
            template_count +=1

           # k = k + 10

            #temp_template = temp_template - temp_template.mean()
            #plt.plot(temp_template,'m')
            
        k+=1
        
    
    # plt.plot(segment_to_clean,'b') 
    
    print(str(template_count) + ' template segments found')
    
    if template_count > 1:
        
        average_template = template_matrix[0:template_count,:].mean(axis=0)
    
        # plt.plot(average_template,'r')
    
        clean_segment = segment_to_clean - average_template
        
        clean_segments_matrix[clean_seg_count,:] = clean_segment
        
        clean_seg_count += 1
    
    seg_count += 1
    
    # clean_segment = clean_segment- clean_segment.mean()
    # plt.plot(clean_segment,'g')
    
    # plt.plot(one_cycle_SSVEP,'b')
    
print(' ')
print(str(clean_seg_count) + ' clean segments')
    
clean_SSVEP = clean_segments_matrix[0:clean_seg_count,:].mean(axis=0)

clean_SSVEP = clean_SSVEP - clean_SSVEP.mean()

plt.plot(clean_SSVEP, label = 'range ' + str(range_cutoff) + ' ' + str(time_range_to_search/1000) + ' seconds search')
    
plt.legend()    
    






########### random walk template construction #################
        



##### random walk #########






# clean_segments_matrix = np.zeros([len(triggers),period])

# clean_seg_count = 0


# for trigger in triggers:

#     segment_to_clean = data[trigger:trigger+period] 
    
    
#     walk_step = 50
    
#     range_of_influence = 10
    
#     num_replications = 100
    
#     all_random_walks = np.zeros([num_replications,period])
    
#     for replication in range(0,num_replications):
        
#         value = segment_to_clean[0] + (random.random() * walk_step * random.choice([-1,1]))
        
#         time_series = []
    
#         for t in range(0,period):
        
#             time_series.append(value)
            
#             if value - segment_to_clean[t] > range_of_influence:
#                 change = -walk_step
#             elif value - segment_to_clean[t] < -range_of_influence:
#                 change = walk_step
#             else:
#                 change = random.choice([walk_step, -walk_step])
            
#             value = value + change
    
#         all_random_walks[replication,:] = time_series
        
#         plt.plot(time_series,'c')
    
#     average_random_walk = all_random_walks.mean(axis=0)
    
#     clean_segment = segment_to_clean - average_random_walk
    
#     clean_segments_matrix[clean_seg_count,:] = clean_segment
    
#     clean_seg_count += 1

#     plt.plot(segment_to_clean,'b')
#     plt.plot(average_random_walk,'m')
#     plt.plot(clean_segment,'g')

    
# clean_SSVEP = clean_segments_matrix[0:clean_seg_count,:].mean(axis=0)

# clean_SSVEP = clean_SSVEP - clean_SSVEP.mean()

# plt.plot(clean_SSVEP)
    




####################  adaptive template subtraction with random segments from the rest of the data  ######################

# num_cycles_for_template = 10
# num_templates = 20

# clean_SSVEP = functions.make_SSVEP_artefact_removal(data, triggers, period, num_cycles_for_template, num_templates)

# plt.plot(clean_SSVEP,'g')



################## template subjetaction with paired segments

# clean_SSVEP = functions.make_SSVEP_artefact_removal_paired_segments(data, triggers, period)

# plt.plot(clean_SSVEP,'g')





# range_cutoff = 900
 
 
# all_segments_matrix = np.zeros([len(triggers), period]) # empty matrix to put segments into
 
# seg_count = 0 # keep track of the number of segments
 
#  # loop through all triggers and put the corresponding segment of data into the matrix
# for trigger in triggers:
 
#      # select a segment of data the lenght of the flicker period, starting from the trigger time 
#     segment =  data[trigger:trigger+period] 
     
#     all_segments_matrix[seg_count,:] = segment
     
#     seg_count += 1
      
    
    
    
# segments_to_use = [] # list 
# segments_not_yet_used = list(range(0,len(triggers)))


# paired_segments_matrix = np.zeros([len(triggers),period])
# paired_seg_count = 0

# # for each segment, try and find a segment with the lowest difference, e.g. the range of the summed wave closest to 0
# for seg_count in range(0,len(triggers)):
     
#     print('Segment ' + str(seg_count) + ' of ' + str(len(triggers)))
     
#     if seg_count in segments_not_yet_used:
         
#         segment = all_segments_matrix[seg_count,:]
         
#         range_of_sum_scores = np.zeros([len(triggers),]) + 10000 # set all to arbitary high value first
         
#         for seg_num in segments_not_yet_used:
             
#             test_segment = all_segments_matrix[seg_num,:]
            
#             average_test_and_segment =  (segment + test_segment) /2
            
#             range_of_sum_scores[seg_num] = np.ptp(average_test_and_segment)

    
#         print(min(range_of_sum_scores)) 
        
#         best_seg_num = np.argmin(range_of_sum_scores)
        
#         best_segment = all_segments_matrix[best_seg_num,:]
        
#         average_pair = (segment + best_segment) /2
        
        
        
#         if (min(range_of_sum_scores) < range_cutoff) and (best_seg_num != seg_count):
 
#             segments_to_use.append(seg_count)
#             segments_to_use.append(best_seg_num)
             
#             segments_not_yet_used.remove(seg_count)
#             segments_not_yet_used.remove(best_seg_num)
         
#             paired_segments_matrix[paired_seg_count,:] = average_pair
#             paired_seg_count += 1


# print(' ')          
# print('Used ' + str(len(segments_to_use)) + ' of ' + str(len(triggers)) + ' triggers')          
 
 
# clean_SSVEP = all_segments_matrix[segments_to_use,:].mean(axis=0) # average to make SSVEP
 
# clean_SSVEP = clean_SSVEP - clean_SSVEP.mean() # baseline correct



# plt.plot(clean_SSVEP,'g')



# paired_SSVEP = paired_segments_matrix[0:paired_seg_count,:].mean(axis=0)

# paired_SSVEP = paired_SSVEP - paired_SSVEP.mean()

# plt.plot(paired_SSVEP,'k')


# average_of_remaining_segment = all_segments_matrix[segments_not_yet_used,:].mean(axis=0) # average to make SSVEP
# average_of_remaining_segment = average_of_remaining_segment - average_of_remaining_segment.mean()

# plt.plot(average_of_remaining_segment)


##################### test with just subtracting random segments

# numbers_of_segs_to_test = np.arange(10,100,10)

# differences_in_amplitude = np.zeros([len(numbers_of_segs_to_test),])

# count = 0


# for number_of_segs in numbers_of_segs_to_test:

# number_of_segs = 2   

# segment_matrix = np.zeros([len(simulated_triggers),period])
# seg_count = 0


# for trigger in simulated_triggers:
    
#     segment = simulated_SSVEP_data[trigger:trigger+period]
    
#     random_segment_matrix = np.zeros([number_of_segs,period])
#     for random_seg_count in range(0,number_of_segs):
#         random_position = random.randint(0, len(simulated_SSVEP_data)-period) # pick a randon point in the data
#         random_segment = simulated_SSVEP_data[random_position:random_position+period]
#         random_segment_matrix[random_seg_count,:] = random_segment
    
#     averaged_random_segs = random_segment_matrix.mean(axis=0)
    
#     segment = segment - averaged_random_segs
    
#     segment_matrix[seg_count,:] = segment
#     seg_count += 1
    
# SSVEP = segment_matrix.mean(axis=0)

# # differences_in_amplitude[count] = np.ptp(SSVEP) - np.ptp(one_cycle_SSVEP)

# # count += 1

# plt.plot(SSVEP)

# plt.plot(one_cycle_SSVEP)
    
        
# plt.plot(numbers_of_segs_to_test,differences_in_amplitude)   
    


