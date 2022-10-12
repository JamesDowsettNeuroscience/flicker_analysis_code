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

trial_time = 30 # total time in seconds

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




########### random walk template construction #################
        
# random_walk_template = np.zeros([len(data)],)

# k = 0

# value = 0

# while k < len(data):
    
#     random_walk_template[k] = data[k] + ((random.random() * 10) - 5)
    
#     k+=1



####################  adaptive template subtraction with random segments from the rest of the data  ######################




length_artefact_segment = period * 5

num_templates = 20

clean_segment_matrix = np.zeros([len(triggers),period])

k = 0
trig_count = 0

while trig_count < len(triggers) - 10:
    
    print('Trigger ' + str(trig_count) + ' of ' + str(len(triggers)))
    
    artefact_segment_start_time = triggers[trig_count]
    
    artefact_segment = data[artefact_segment_start_time:artefact_segment_start_time+length_artefact_segment]

    template_matrix = np.zeros([num_templates,length_artefact_segment])
        

    template_count = 0
    k = period
    
    while (template_count < num_templates) and (k < len(data)-length_artefact_segment):
 
        temp_template = data[k:k+length_artefact_segment]

        if np.corrcoef(artefact_segment,temp_template)[0,1] > 0.9:
           
            
            ## check up to 20 data points ahead and behind for the best correlation
            temp_corr_scores = np.zeros([40,])
            
            count = 0
            for t in range(-20,20):
                temp_template = data[k+t:k+t+length_artefact_segment]
              #  plt.plot(temp_template,'c')
                temp_corr_scores[count] = np.corrcoef(artefact_segment,temp_template)[0,1]
                count += 1
                
            best_time_index = np.argmax(temp_corr_scores) - 20
            
            best_template = data[k+best_time_index:k+best_time_index+length_artefact_segment]
            
            template_matrix[template_count,:] = best_template
            template_count += 1
        
            best_template = best_template - best_template.mean()
            
          #  plt.plot(best_template,'r')
            
            k = k + 500
            
        k += 1

    average_template = template_matrix[0:template_count,:].mean(axis=0)

  #  plt.plot(artefact_segment,'r')

    cleaned_artefact_segment = artefact_segment - average_template
    
 #   plt.plot(cleaned_artefact_segment,'g')


    ## put the 9 segments from the one second segment into the clean segment matrix
    trigger_time = artefact_segment_start_time
    while trigger_time <= (artefact_segment_start_time + length_artefact_segment - period):
        
        segment = cleaned_artefact_segment[trigger_time - artefact_segment_start_time:trigger_time - artefact_segment_start_time+period]
        
        clean_segment_matrix[trig_count,:] = segment
        
        trig_count += 1
        
        trigger_time = triggers[trig_count] 
        

clean_SSVEP = clean_segment_matrix[0:trig_count,:].mean(axis=0) # average segments to make the SSVEP

clean_SSVEP = clean_SSVEP - clean_SSVEP.mean() # baseline correct

plt.plot(clean_SSVEP,'g')




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
    


