#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 14:25:54 2022

@author: James Dowsett
"""

### test split SSVEP function on random noise. This will output the correlation above which is significant at 0.05 (one tailed)

import numpy as np

from flicker_analysis_package import functions

import matplotlib.pyplot as plt

import math

sample_rate = 1000
 
period = 25
 
simulated_SSVEP_data = np.zeros([60 * sample_rate])
simulated_triggers = np.arange(0, len(simulated_SSVEP_data)-period, period) # make triggers, stop one period length before the end
 
noise_amplitude = 1

num_trials = 10

sig_corr_scores = np.zeros(num_trials,)

for trial in range(0,num_trials):
     
    num_loops = 1000
    corr_scores = np.zeros([num_loops,])
     
    # noise = np.random.rand(len(simulated_SSVEP_data),) * noise_amplitude
     
    for loop in range(0,num_loops):
     
        noise = np.random.rand(len(simulated_SSVEP_data),) * noise_amplitude    
        
        split_SSVEPs = functions.compare_SSVEPs_split(noise, simulated_triggers, period)
        
        SSVEP_1 = split_SSVEPs[0]
        SSVEP_2 = split_SSVEPs[1]
        
        corr_scores[loop] = np.corrcoef(SSVEP_1, SSVEP_2)[0,1]
    
         
    #plt.hist(corr_scores,50)
    
    significant_Z_score = 1.645
    
    # corr_scores_as_Z = corr_scores/np.std(corr_scores) # convert to Z scores
    # plt.hist(corr_scores_as_Z,50)

    # count = (corr_scores > 0.336).sum()
    
    # p = count/num_loops 
    # print(p)
    # Z_score = (0.336 - corr_scores.mean()) / np.std(corr_scores)


        

    sig_corr = (significant_Z_score * np.std(corr_scores)) + corr_scores.mean()      
    
   
    
    print('significant correlation = ' + str(sig_corr))
    
    sig_corr_scores[trial] = sig_corr
    
    
    
average_significant_correlation = sig_corr_scores.mean()

print('  ')
print('Average significant correlation = ' + str(average_significant_correlation))


print('  ')

for correlation in(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    Z_score = (correlation - corr_scores.mean()) / np.std(corr_scores)
    count = (corr_scores > correlation).sum()
    p = count/num_loops 
    print('correlation: ' + str(correlation) + ' = Z-score of ' + str(np.round(Z_score,2)) + ', p = ' + str(p))



