#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 12:19:03 2022

@author: James Dowsett
"""

from flicker_analysis_package import functions

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats


### information about the experiment: Gamma walk 1

path = '/home/james/Active_projects/Gamma_walk/Gamma_walking_experiment_1/raw_data_for_analysis_package/'

electrode_names = ('VEOG', 'blink', 'P3', 'P4', 'O2', 'Pz', 'O1', 'HEOG', 'x_dir', 'y_dir', 'z_dir')

condition_names = ('walking', 'standing')

sample_rate = 1000

num_subjects = 10

frequencies_to_use = (30, 35, 40, 45, 50, 55)

for subject in range(1,num_subjects+1):
    
    print(' ')
    print('Subject ' + str(subject))
    
    plt.figure()
    plt.suptitle(subject)
    
    electrode = 6
    
    plot_count = 1
    for frequency in frequencies_to_use:
        
        print(str(frequency) + ' Hz')
        plt.subplot(2,3,plot_count)
        plt.title(str(frequency) + ' Hz')

        for condition in (0, 1):
            
            condition_name = condition_names[condition]
            
            data_file_name = 'subject_' + str(subject) + '_electrode_' + str(electrode) + '_data.npy'
            
            data = np.load(path + data_file_name)
            
            triggers_file_name = 'subject_' + str(subject) + '_' + condition_name + '_' + str(frequency) + 'Hz_triggers.npy'
            
            triggers = np.load(path + triggers_file_name)
            
            
            
            ### make SSVEP
            
            period = int(np.round(sample_rate/frequency))
            
            SSVEP = functions.make_SSVEPs(data, triggers, period)

          #  SNR = functions.SNR_random(data, triggers, period)
            
            # random split into two SSVEPs
            split_SSVEPs = functions.compare_SSVEPs_split(data, triggers, period)
            
            SSVEP_1 = split_SSVEPs[0]
            SSVEP_2 = split_SSVEPs[1]
            
            ## plots
            if condition == 0: # walking
                plt.plot(SSVEP, 'r')#, label = (condition_name + ' ' + str(np.round(SNR,2))))
                plt.plot(SSVEP_1,'m')
                plt.plot(SSVEP_2,'m')
            elif condition ==  1: # standing
                plt.plot(SSVEP, 'b')#, label = (condition_name + ' ' + str(np.round(SNR,2))))
                plt.plot(SSVEP_1,'c')
                plt.plot(SSVEP_2,'c')
                
           # print(condition_name + '  SNR = ' + str(np.round(SNR,2)))
            
           

        
            # num_loops = 1000
            # Z_score = functions.make_SSVEPs_random(data, triggers, period, num_loops)
            
            # print(condition_name + ' Z score =  ' + str(Z_score))
            
            # length = 10
            # evoked_FFT = functions.evoked_fft(data, triggers, length, sample_rate)
            
            # time_vector = np.linspace(0, sample_rate, num=int(length * sample_rate))
            # plt.plot(time_vector, evoked_FFT, label = condition_name)
            # plt.xlim(0, 100)
            
       # plt.legend()
        plot_count += 1