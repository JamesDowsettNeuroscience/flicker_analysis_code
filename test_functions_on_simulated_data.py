#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 12:35:34 2022

@author: James Dowsett
"""

from flicker_analysis_package import functions

import numpy as np
import matplotlib.pyplot as plt


file_names = ('simulated_data_40Hz_Consistent_Amplitude', 'simulated_data_40Hz_Consistent_smaller_Amplitude')

for condition in(0,1):
    
    ## load simulated data and triggers
    
    data = np.load(file_names[condition] + '.npy')
    
    triggers = np.load('simulated_triggers_40Hz.npy')
    
    
    #plt.plot(data) # plot before filter
    
    ## filter
    data = functions.high_pass_filter(5000, data)
    
    
    #plt.plot(data) # plot after filter
    
    
    ## make SSVEP
    
    SSVEP = functions.make_SSVEPs(data, triggers, 125)
    
    plt.plot(SSVEP)
    
    num_loops = 100
    
   # functions.make_SSVEPs_random(data, triggers, 125, num_loops)