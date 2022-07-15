#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 16:38:22 2022

@author: James Dowsett
"""


#########  Analysis of gamma flicker walk experiment 2   ####################



from flicker_analysis_package import functions

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats
from random import choice
import statistics


### information about the experiment: Gamma walk 1

path = '/home/james/Active_projects/Gamma_walk/gamma_walk_experiment_2/Right_handed_participants/raw_data_for_analysis_package/'

electrode_names = np.load(path + '/electrode_names.npy')

condition_labels = ['W', 'S', 'black']

condition_names = ['Walking', 'Standing', 'Blackout']

sample_rate = 1000

num_subjects = 24

frequencies_to_use = (35, 40)


trig_1_times = [-1, -1, -1, -1, -1, -1]
trig_2_times = [15, 13, 11, 10, 9, 8]
trig_length = 4

#######################################


laplacian = 1

if laplacian == 0:
    montage_name = 'Cz Reference'
elif laplacian == 1:
    montage_name = 'Laplacian'

### Matrices to store results and SSVEPs


all_SSVEPs = np.zeros([num_subjects,2,64,3,29]) # subject, frequency, electrode, condition, SSVEP data (29 data points is the largest SSVEP, 35 Hz)

SSVEP_amplitudes = np.zeros([num_subjects,2,64,3]) # subject, frequency, electrode , condition

SSVEP_walking_standing_correlations = np.zeros([num_subjects,2,64]) # subject, frequency, electrode

SSVEP_phase_scores = np.zeros([num_subjects,6,8])




##################################



for subject in range(1,num_subjects+1):
   
    print('  ')
    print('Subject ' + str(subject))
    print(' ')
 
    conditions_to_use = np.arange(0,3)
    
    if subject == 1: # subject one had no blackout condition
        conditions_to_use = np.array([0, 1])

    
    for frequency_count in range(0,2):
        
        plt.figure()
        plt.suptitle('Subject ' + str(subject) + '  ' + str(frequencies_to_use[frequency_count]) + ' Hz ' + montage_name)
        
        frequency = frequencies_to_use[frequency_count]
        
        for condition_count in conditions_to_use:

            condition_name = condition_labels[condition_count] + str(frequencies_to_use[frequency_count])
            

            
            file_name = 'S' + str(subject) + '_' + condition_name
    
            ## load raw data
            if laplacian == 0:
                all_data = np.load(path + file_name + '_all_data_interpolated.npy')
            elif laplacian == 1:
                all_data = np.load(path + file_name + '_all_data_interpolated_laplacian.npy')
            
            ## load triggers
            triggers = np.load(path + file_name + '_all_triggers.npy')
         
            for electrode in range(0,64):
                
                electrode_name = electrode_names[electrode]
                
                # print(' ')
                # print(electrode_name)
                # print(' ')
                
                data = all_data[electrode,:]
              
        
                    ### make SSVEP
            
                period = int(np.round(sample_rate/frequency))
                
                SSVEP = functions.make_SSVEPs(data, triggers, period)
                
                
                # put the SSVEP into the matrix
                all_SSVEPs[subject-1,frequency_count,electrode,condition_count,0:period] = SSVEP # subject, frequency, electrode, condition, SSVEP data (29 data points is the largest SSVEP, 35 Hz)
    
                # record the SSVEP amplitudes
                SSVEP_amplitudes[subject-1,frequency_count,electrode,condition_count] = np.ptp(SSVEP) # subject, frequency, electrode , condition
                
                plt.subplot(8,8,electrode+1)
                
                plt.title(electrode_name)
                
                plt.plot(SSVEP)
              
            # set constant y axis in plots
            max_amplitude = np.max(SSVEP_amplitudes[subject-1,frequency_count,:,:]) 
            for electrode in range(0,64):  
                plt.subplot(8,8,electrode+1)
                plt.ylim([-(max_amplitude/2), (max_amplitude/2)])
         