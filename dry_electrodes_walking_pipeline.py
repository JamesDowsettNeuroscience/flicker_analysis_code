#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 16:17:43 2022

@author: James Dowsett
"""

#########  Analysis of Dry electrode walking experiment  ####################



from flicker_analysis_package import functions

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats
from random import choice
import statistics


path = '/home/james/Active_projects/mentalab_dry_electrodes/mentalab_test/subject_data/'

electrode_names = ('O1', 'O2', 'P3', 'P4', 'P7', 'P8', 'HEOG', 'VEOG', 'x_dir', 'y_dir', 'z_dir')

condition_names = ('sigi_stand', 'sigi_walk', 'flicker_stand', 'flicker_walk', 'blackout')

location_names = ('hall', 'lobby')

sample_rate = 1000

num_subjects = 10

period = 25




for subject in range(1,2):
    
    print(subject)
    
    plt.figure()
    plt.title(subject)
    
    electrode = 1
    
    # load data, all conditions should be one data file
    
    file_name = path + 'subject_' + str(subject+1) + '/subject_' + str(subject+1) + '_chan_' + str(electrode) + '_data.npy'
    
    data = np.load(file_name)
    
    plt.plot(data)
    
    
    # get Accelerometer data
    
    accelerometer_x_file_name = path + 'subject_' + str(subject+1) + '/subject_' + str(subject+1) + '_acc_chan_1_data.npy'  
    
    accelerometer_x_data = np.load(accelerometer_x_file_name)
        
    
    accelerometer_y_file_name = path + 'subject_' + str(subject+1) + '/subject_' + str(subject+1) + '_acc_chan_2_data.npy'  
    
    accelerometer_y_data = np.load(accelerometer_y_file_name)
        
    
    accelerometer_z_file_name = path + 'subject_' + str(subject+1) + '/subject_' + str(subject+1) + '_acc_chan_3_data.npy'  
    
    accelerometer_z_data = np.load(accelerometer_z_file_name)
      
    plot_scale = 1000
    plt.plot(accelerometer_x_data*plot_scale)
    plt.plot(accelerometer_y_data*plot_scale)
    plt.plot(accelerometer_z_data*plot_scale)
    
    # get gyroscope data

    gyroscope_x_file_name = path + 'subject_' + str(subject+1) + '/subject_' + str(subject+1) + '_acc_chan_4_data.npy'  
    
    gyroscope_x_data = np.load(gyroscope_x_file_name)
    
    
    gyroscope_y_file_name = path + 'subject_' + str(subject+1) + '/subject_' + str(subject+1) + '_acc_chan_6_data.npy'  
    
    gyroscope_y_data = np.load(gyroscope_y_file_name)
        
    
    gyroscope_z_file_name = path + 'subject_' + str(subject+1) + '/subject_' + str(subject+1) + '_acc_chan_5_data.npy'  
    
    gyroscope_z_data = np.load(gyroscope_z_file_name)    
        

 
  #  plt.plot(gyroscope_x_data)
    plt.plot(gyroscope_y_data)
    plt.plot(gyroscope_z_data)
    
    
    
    for condition in range(0,5):
        
        ## load triggers for condition
        
        condition_name = condition_names[condition]
        
        trigger_file_name = path + 'subject_' + str(subject+1) + '/Subject_' + str(subject+1) + '_all_triggers_' + condition_name + '.npy'
        
        triggers = np.load(trigger_file_name)
        
        trigger_time_series = np.zeros([len(data)],)
        
        for trigger in triggers:
            trigger_time_series[trigger] = -1000000
            
        plt.plot(trigger_time_series)
            
            
    
    
    