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

condition_names = ['W35', 'S35', 'W40', 'S40', 'black35', 'black40']

sample_rate = 1000

num_subjects = 24

frequencies_to_use = (35, 40)


trig_1_times = [-1, -1, -1, -1, -1, -1]
trig_2_times = [15, 13, 11, 10, 9, 8]
trig_length = 4

#######################################


laplacian = 1

### Matrices to store results and SSVEPs


all_SSVEPs = np.zeros([num_subjects,2,64,2,29]) # subject, frequency, electrode, condition, SSVEP data (29 data points is the largest SSVEP, 35 Hz)

SSVEP_amplitudes = np.zeros([num_subjects,2,64,2]) # subject, frequency, electrode , condition

SSVEP_walking_standing_correlations = np.zeros([num_subjects,2,64]) # subject, frequency, electrode

SSVEP_phase_scores = np.zeros([num_subjects,6,8])


blackout_SSVEPs = np.zeros([num_subjects,64,2,29]) # blackout  (29 data points is the largest SSVEP, 35 Hz)

blackout_amplitudes = np.zeros([num_subjects,64,2])

##################################



for subject in range(1,num_subjects+1):
   
    print('  ')
    print('Subject ' + str(subject))
    print(' ')
 
    conditions_to_use = np.arange(0,6)
    
    if subject == 1: # subject one had no blackout condition
        conditions_to_use = np.array([0, 1, 2, 3])

    for condition in conditions_to_use:
        
        if condition == 0 or condition == 1 or condition == 4:
            frequency = 35
           
            plt.figure(2)
            plt.suptitle('35 Hz')
            
        elif condition == 2 or condition == 3 or condition == 5:
        
            frequency = 40
            
            plt.figure(3)
            plt.suptitle('40 Hz')

                  
        condition_name = condition_names[condition]
        
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
            
            print(' ')
            print(electrode_name)
            print(' ')
            
            data = all_data[electrode,:]
          
    
                ### make SSVEP
        
            period = int(np.round(sample_rate/frequency))
            
            SSVEP = functions.make_SSVEPs(data, triggers, period)
            
            
            plt.subplot(8,8,electrode+1)
            
            plt.title(electrode_name)
            
            plt.plot(SSVEP)
            
            plt.ylim([-0.0005, 0.0005])
            