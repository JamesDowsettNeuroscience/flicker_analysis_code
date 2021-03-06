#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 12:55:09 2022

@author: James Dowsett
"""

#########  Analysis of gamma flicker walk experiment 1   ####################



from flicker_analysis_package import functions

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats
from random import choice
import statistics


### information about the experiment: Gamma walk 1

path = '/home/james/Active_projects/Gamma_walk/Gamma_walking_experiment_1/raw_data_for_analysis_package/'

electrode_names = ('VEOG', 'blink', 'P3', 'P4', 'O2', 'Pz', 'O1', 'HEOG', 'x_dir', 'y_dir', 'z_dir')

condition_names = ('standing', 'walking')

sample_rate = 1000

num_subjects = 10

frequencies_to_use = (30, 35, 40, 45, 50, 55)


trig_1_times = [-1, -1, -1, -1, -1, -1]
trig_2_times = [15, 13, 11, 10, 9, 8]
trig_length = 4

#######################################


### Matrices to store results and SSVEPs

SIGI_SSVEPs = np.zeros([num_subjects,6,8,2,25]) # subject, frequency, electrode, condition, SSVEP data (SIGI always 40 Hz, lenght = 25)

SIGI_amplitudes = np.zeros([num_subjects,6,8,2]) # subject, frequency, electrode, condition

SIGI_walking_standing_correlations = np.zeros([num_subjects,6,8])

SIGI_phase_scores = np.zeros([num_subjects,6,8])


all_SSVEPs = np.zeros([num_subjects,6,8,2,34]) # subject, frequency, electrode, condition, SSVEP data (34 data points is the largest SSVEP)

SSVEP_amplitudes = np.zeros([num_subjects,6,8,2]) # subject, frequency, electrode , condition

SSVEP_walking_standing_correlations = np.zeros([num_subjects,6,8]) # subject, frequency, electrode

SSVEP_phase_scores = np.zeros([num_subjects,6,8])


blackout_SSVEPs = np.zeros([num_subjects,8,25]) # blackout was 40 Hz, so length = 25

blackout_amplitudes = np.zeros([num_subjects,8])

##################################

for subject in range(1,11):
   
    print('  ')
    print('Subject ' + str(subject))
    print(' ')
 
    
    for electrode in range(0,8):
        
        electrode_name = electrode_names[electrode]
        
        print(' ')
        print(electrode_name)
        print(' ')
        
        ## load raw data
        
        data_file_name = 'subject_' + str(subject) + '_electrode_' + str(electrode) + '_data.npy'
        
        raw_data = np.load(path + data_file_name)                



        ####### SIGI conditions
        
        frequency_count = 0
        for frequency in frequencies_to_use: # loop for each frequency to match the number of segments from each frequency 

            for condition in range(0,2):
                
                ## load triggers from real SSVEP condition to match the number of triggers to use
                triggers_file_name = 'subject_' + str(subject) + '_' + condition_names[condition] + '_' + str(frequency) + 'Hz_triggers.npy'   
                triggers = np.load(path + triggers_file_name)    
            
                num_triggers_to_use = len(triggers)
                
                
                # load the SIGI triggers
                triggers = np.load(path + 'subject_' + str(subject) + '_SIGI_' + condition_names[condition] + '_triggers.npy')
                
                # only use the same number of triggers that there were in the real SSVEP condition
                triggers = triggers[0:num_triggers_to_use]
                
                print(condition_names[condition] + ' ' + str(len(triggers)))
                
                ### make SSVEP
                
                period = int(np.round(sample_rate/40))
                
                SSVEP = functions.make_SSVEPs(raw_data, triggers, 25) # SIGI was always 40 Hz, length = 25

                # plt.plot(SSVEP)
        
                SIGI_amplitudes[subject-1,frequency_count,electrode,condition] = np.ptp(SSVEP) # save amplitude
                
                SIGI_SSVEPs[subject-1,frequency_count,electrode,condition,:] = SSVEP # save the SSVEP
                
                    # make a copy to later compare walking and standing
                if condition == 0:
                    standing_SSVEP = np.copy(SSVEP)
                elif condition== 1:
                    walking_SSVEP = np.copy(SSVEP)
        
        
            # get walking/standing correlations and phase shift
            SIGI_walking_standing_correlations[subject-1,frequency_count,electrode] = np.corrcoef(standing_SSVEP, walking_SSVEP)[0,1]
        
            SIGI_phase_scores[subject-1,frequency_count,electrode] = functions.cross_correlation(standing_SSVEP, walking_SSVEP)
        
            frequency_count += 1
            
            
    
        ######### make real SSVEPs  ########################
        frequency_count = 0
        for frequency in frequencies_to_use:
            for condition in range(0,2):
            
            
                ## load triggers
                triggers_file_name = 'subject_' + str(subject) + '_' + condition_names[condition] + '_' + str(frequency) + 'Hz_triggers.npy'
                
                triggers = np.load(path + triggers_file_name)
                
                ### linear interpolation
                data_linear_interpolation = functions.linear_interpolation(raw_data, triggers, trig_1_times[frequency_count], trig_2_times[frequency_count], trig_length)
                
                
                
                ### make SSVEP
                
                period = int(np.round(sample_rate/frequency))
                
                SSVEP = functions.make_SSVEPs(data_linear_interpolation, triggers, period)

                # save amplitude
                SSVEP_amplitudes[subject-1,frequency_count,electrode,condition] = np.ptp(SSVEP)
                
                all_SSVEPs[subject-1,frequency_count,electrode,condition,0:len(SSVEP)] = SSVEP # save the SSVEP
                
                # make a copy to later compare walking and standing
                if condition == 0:
                    standing_SSVEP = np.copy(SSVEP)
                elif condition== 1:
                    walking_SSVEP = np.copy(SSVEP)
                    
            # save correlations and phase shift
            SSVEP_walking_standing_correlations[subject-1,frequency_count,electrode] = np.corrcoef(standing_SSVEP, walking_SSVEP)[0,1]
               
            SSVEP_phase_scores[subject-1,frequency_count,electrode] = functions.cross_correlation(standing_SSVEP, walking_SSVEP)

            frequency_count += 1
                    
                
    ############# make blackout SSVEPs  ######################
    
   
        
        ## load triggers
        triggers_file_name = 'subject_' + str(subject) + '_blackout_triggers.npy'
        
        triggers = np.load(path + triggers_file_name)
        
        ### linear interpolation, use 40 Hz trigger times, = frequency 2
        data_linear_interpolation = functions.linear_interpolation(raw_data, triggers, trig_1_times[2], trig_2_times[2], trig_length)
        
         
        period = int(np.round(sample_rate/40))
        
        SSVEP = functions.make_SSVEPs(data_linear_interpolation, triggers, period)
        
        blackout_amplitudes[subject-1,electrode] = np.ptp(SSVEP)

        blackout_SSVEPs[subject-1,electrode,:] = SSVEP # save the SSVEP



######  plots

## check raw SSVEPs for each electrode

electrode = 1 #('VEOG', 'blink', 'P3', 'P4', 'O2', 'Pz', 'O1', 'HEOG', 'x_dir', 'y_dir', 'z_dir')

for subject in range(1,11):
    
    plt.figure()
    plt.suptitle('Subject ' + str(subject) + ' ' + electrode_names[electrode])
    
    for frequency_count in range(0,6):
        
        plt.subplot(3,3,frequency_count+1)
        
        plt.title(str(frequencies_to_use[frequency_count]) + ' Hz')
        
        period = int(np.round(sample_rate/frequencies_to_use[frequency_count]))
        
        standing_SSVEP = all_SSVEPs[subject-1,frequency_count,electrode,0,0:period]
        walking_SSVEP = all_SSVEPs[subject-1,frequency_count,electrode,1,0:period]
        
        plt.plot(standing_SSVEP,'b')
        plt.plot(walking_SSVEP,'r')

    
    plt.subplot(3,3,3)

    blackout_SSVEP =  blackout_SSVEPs[subject-1,electrode,:] 
    
    plt.plot(blackout_SSVEP,'k')


    plt.subplot(3,3,8)
    for frequency_count in range(0,6):
        
        standing_SIGI = SIGI_SSVEPs[subject-1,frequency_count,electrode,0,:]
        walking_SIGI = SIGI_SSVEPs[subject-1,frequency_count,electrode,1,:]

        plt.plot(standing_SIGI,'b')
        plt.plot(walking_SIGI,'r')




###########  plot amplitudes

electrodes_to_use = (2,3,5,6,4) #  ('VEOG', 'blink', 'P3', 'P4', 'O2', 'Pz', 'O1', 'HEOG', 'x_dir', 'y_dir', 'z_dir')
electrodes_used_names = ('P3', 'P4', 'Pz', 'O1', 'O2')


plt.figure()
plt.subplot(2,1,1)
plt.title('All Amplitudes')

small_dot_size = 2

electrode_count = 0

for electrode in electrodes_to_use:  #
    
    for frequency_count in range(0,6):

        # standing
        all_subjects_amplitudes_standing = SSVEP_amplitudes[:,frequency_count,electrode,0]

        plt.scatter(np.zeros([10,])+((electrode_count*10) + frequency_count),all_subjects_amplitudes_standing, c='b', s=small_dot_size)

        mean_amplitude_standing = all_subjects_amplitudes_standing.mean()
        
        std_error_amplitude_standing = np.std(all_subjects_amplitudes_standing) / math.sqrt(10)

        plt.errorbar((electrode_count*10) + frequency_count, mean_amplitude_standing,yerr = std_error_amplitude_standing, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'b', ecolor='b')  

        # walking
        all_subjects_amplitudes_walking = SSVEP_amplitudes[:,frequency_count,electrode,1]
    
        plt.scatter(np.zeros([10,])+((electrode_count*10) + frequency_count+0.5),all_subjects_amplitudes_walking, c='r', s=small_dot_size)

        mean_amplitude_walking = all_subjects_amplitudes_walking.mean()
        
        std_error_amplitude_walking = np.std(all_subjects_amplitudes_walking) / math.sqrt(10)
        
        plt.errorbar((electrode_count*10) + frequency_count + 0.5, mean_amplitude_walking,yerr = std_error_amplitude_walking, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'r', ecolor='r')  
     
    if electrode == 5: # electrode Pz is the only electrode where the signal generator is approximatly the same size as real SSVEPs
         #SIGI
        SIGI_amplitudes_standing = SIGI_amplitudes[:,frequency_count,electrode,0]  
        
        plt.scatter(np.zeros([10,])+((electrode_count*10) + 6),SIGI_amplitudes_standing, c='c', s=small_dot_size)
          
        mean_SIGI_standing = SIGI_amplitudes_standing.mean()
        
        std_error_SIGI_standing = np.std(SIGI_amplitudes_standing) / math.sqrt(10)
        
        plt.errorbar((electrode_count*10) + 6, mean_SIGI_standing,yerr = std_error_SIGI_standing, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'c', ecolor='c')  
    
        SIGI_amplitudes_walking = SIGI_amplitudes[:,frequency_count,electrode,1]  
        
        plt.scatter(np.zeros([10,])+((electrode_count*10) + 6.5),SIGI_amplitudes_walking, c='m', s=small_dot_size)
          
        mean_SIGI_walking = SIGI_amplitudes_walking.mean()
        
        std_error_SIGI_walking = np.std(SIGI_amplitudes_walking) / math.sqrt(10)
        
        plt.errorbar((electrode_count*10) + 6.5, mean_SIGI_walking,yerr = std_error_SIGI_walking, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'm', ecolor='m')  
      

    # blackout
    all_subjects_blackout =  blackout_amplitudes[:,electrode]   

    plt.scatter(np.zeros([10,])+(electrode_count*10) + 7,all_subjects_blackout, c='k', s=small_dot_size)

    mean_amplitude_blackout = np.nanmean(all_subjects_blackout)

    std_error_blackout = np.nanstd(all_subjects_blackout) / math.sqrt(10)
    
    plt.errorbar((electrode_count*10) + 7, mean_amplitude_blackout,yerr = std_error_blackout, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'k', ecolor='k')  
    
    
    
    electrode_count += 1
    

plt.errorbar(-100, 0,yerr = 0, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'b', ecolor='b', label = ' Standing') 
plt.errorbar(-100, 0,yerr = 0, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'r', ecolor='r', label = ' Walking')  
plt.errorbar(-100, 0,yerr = 0, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'c', ecolor='c', label = ' Signal Generator \n Standing') 
plt.errorbar(-100, 0,yerr = 0, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'm', ecolor='m', label = ' Signal Generator \n Walking')  
plt.errorbar(-100, 0,yerr = 0, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'k', ecolor='k', label = ' Blackout')  

plt.xlim(-1, 48)

plt.legend()

plt.ylabel('Peak to Peak Amplitude (\u03BCV)')


    
## set x axis ticks
x = np.arange(0,50,10)+3
y = np.zeros([5,])
labels = electrodes_used_names
plt.xticks(x, labels, rotation='vertical')
plt.show()





#### plot amplitudes just Pz

# frequency_names = ('30 Hz', '35 Hz', '40 Hz', '45 Hz', '50 Hz', '55 Hz', 'Signal \n Generator \n (40 Hz)', 'Blackout \n (40 Hz)')


# plt.figure()
# plt.title('Peak to peak amplitudes Pz')

# electrode = 5

# for frequency_count in range(0,6):

#     # standing
#     all_subjects_amplitudes_standing = SSVEP_amplitudes[:,frequency_count,electrode,0]

#     plt.scatter(np.zeros([10,]) + frequency_count,all_subjects_amplitudes_standing, c='b', s=small_dot_size)

#     mean_amplitude_standing = all_subjects_amplitudes_standing.mean()
    
#     std_error_amplitude_standing = np.std(all_subjects_amplitudes_standing) / math.sqrt(10)

#     plt.errorbar(frequency_count, mean_amplitude_standing,yerr = std_error_amplitude_standing, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'b', ecolor='b')  

#     # walking
#     all_subjects_amplitudes_walking = SSVEP_amplitudes[:,frequency_count,electrode,1]

#     plt.scatter(np.zeros([10,]) + frequency_count+0.2,all_subjects_amplitudes_walking, c='r', s=small_dot_size)

#     mean_amplitude_walking = all_subjects_amplitudes_walking.mean()
    
#     std_error_amplitude_walking = np.std(all_subjects_amplitudes_walking) / math.sqrt(10)
    
#     plt.errorbar(frequency_count + 0.2, mean_amplitude_walking,yerr = std_error_amplitude_walking, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'r', ecolor='r')  
 

# SIGI_amplitudes_standing = SIGI_amplitudes[:,frequency_count,electrode,0]  

# plt.scatter(np.zeros([10,]) + 6,SIGI_amplitudes_standing, c='c', s=small_dot_size)
  
# mean_SIGI_standing = SIGI_amplitudes_standing.mean()

# std_error_SIGI_standing = np.std(SIGI_amplitudes_standing) / math.sqrt(10)

# plt.errorbar(6, mean_SIGI_standing,yerr = std_error_SIGI_standing, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'c', ecolor='c')  

# SIGI_amplitudes_walking = SIGI_amplitudes[:,frequency_count,electrode,1]  

# plt.scatter(np.zeros([10,]) + 6.2,SIGI_amplitudes_walking, c='m', s=small_dot_size)
  
# mean_SIGI_walking = SIGI_amplitudes_walking.mean()

# std_error_SIGI_walking = np.std(SIGI_amplitudes_walking) / math.sqrt(10)

# plt.errorbar(6.2, mean_SIGI_walking,yerr = std_error_SIGI_walking, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'm', ecolor='m')  
  

# # blackout
# all_subjects_blackout =  blackout_amplitudes[:,electrode]   

# plt.scatter(np.zeros([10,]) + 7,all_subjects_blackout, c='k', s=small_dot_size)

# mean_amplitude_blackout = np.nanmean(all_subjects_blackout)

# std_error_blackout = np.nanstd(all_subjects_blackout) / math.sqrt(10)

# plt.errorbar(7, mean_amplitude_blackout,yerr = std_error_blackout, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'k', ecolor='k')  

# plt.ylabel('Peak to Peak Amplitude (\u03BCV)')


# ## set x axis ticks
# x = np.arange(0,8,1)
# y = np.zeros([8,])
# labels = frequency_names
# plt.xticks(x, labels, rotation='vertical')
# plt.show()

# plt.errorbar(-100, 0,yerr = 0, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'b', ecolor='b', label = ' Standing') 
# plt.errorbar(-100, 0,yerr = 0, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'r', ecolor='r', label = ' Walking')  
# plt.errorbar(-100, 0,yerr = 0, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'c', ecolor='c', label = ' Signal Generator \n Standing') 
# plt.errorbar(-100, 0,yerr = 0, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'm', ecolor='m', label = ' Signal Generator \n Walking')  
# plt.xlim(-1, 8)
# plt.legend()




####### plot amplitude differences, electrodes 'P3', 'P4', 'Pz', 'O1', 'O2'


electrodes_to_use = (2,3,5,6,4) #  ('VEOG', 'blink', 'P3', 'P4', 'O2', 'Pz', 'O1', 'HEOG', 'x_dir', 'y_dir', 'z_dir')
electrodes_used_names = ('P3', 'P4', 'Pz', 'O1', 'O2')

#plt.figure()
plt.subplot(2,1,2)
plt.title('Walking minus Standing Amplitude differences')

small_dot_size = 2

colours = ['r','m','g','b','c','y','k']

electrode_count = 0

for electrode in electrodes_to_use:
    
    for frequency_count in range(0,6):

        # standing
        all_subjects_amplitudes_standing = SSVEP_amplitudes[:,frequency_count,electrode,0]

          # walking
        all_subjects_amplitudes_walking = SSVEP_amplitudes[:,frequency_count,electrode,1]
        
        # difference
        amplitude_difference = all_subjects_amplitudes_walking - all_subjects_amplitudes_standing

     #   plt.scatter(np.zeros([10,])+((electrode_count*10) + frequency_count),amplitude_difference, c=colours[frequency_count], s=small_dot_size)

        mean_amplitude_difference = amplitude_difference.mean()
        
        std_error_amplitude_difference = np.std(amplitude_difference) / math.sqrt(10)

        plt.errorbar((electrode_count*10) + frequency_count, mean_amplitude_difference,yerr = std_error_amplitude_difference, solid_capstyle='projecting', capsize=5,  fmt='o', color= colours[frequency_count], ecolor=colours[frequency_count])  

    if electrode == 5:
        ## SIGI
        SIGI_amplitudes_standing = SIGI_amplitudes[:,frequency_count,electrode,0]
        SIGI_amplitudes_walking = SIGI_amplitudes[:,frequency_count,electrode,1]
      
        SIGI_difference = SIGI_amplitudes_walking - SIGI_amplitudes_standing
      
      #  plt.scatter(np.zeros([10,])+((electrode_count*10) + 6),SIGI_difference, c=colours[6], s=small_dot_size)
    
        mean_SIGI_difference = SIGI_difference.mean()
            
        std_error_SIGI_difference = np.std(SIGI_difference) / math.sqrt(10)
            
        plt.errorbar((electrode_count*10) + 6, mean_SIGI_difference,yerr = std_error_SIGI_difference, solid_capstyle='projecting', capsize=5,  fmt='o', color= colours[6], ecolor=colours[6])  
    
    electrode_count += 1    
    
zero_line = np.arange(-5,50)
plt.plot(zero_line,np.zeros([55,]),'k--')

## set x axis ticks
x = np.arange(0,50,10)+3
y = np.zeros([5,])
labels = electrodes_used_names
plt.xticks(x, labels, rotation='vertical')
plt.show()

plt.errorbar(-100, 0,yerr = 0, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'r', ecolor='r', label = ' 30 Hz') 
plt.errorbar(-100, 0,yerr = 0, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'm', ecolor='m', label = ' 35 Hz')  
plt.errorbar(-100, 0,yerr = 0, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'g', ecolor='g', label = ' 40 Hz') 
plt.errorbar(-100, 0,yerr = 0, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'b', ecolor='b', label = ' 45 Hz')  
plt.errorbar(-100, 0,yerr = 0, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'c', ecolor='c', label = ' 50 Hz') 
plt.errorbar(-100, 0,yerr = 0, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'y', ecolor='y', label = ' 55 Hz')  
plt.errorbar(-100, 0,yerr = 0, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'k', ecolor='k', label = ' Signal Generator \n (40 Hz)') 

plt.ylabel('Amplitude difference (\u03BCV)')

plt.xlim(-1, 48)
plt.legend()



##### plot phase shifts

electrodes_to_use = (2,3,5,6,4) #  ('VEOG', 'blink', 'P3', 'P4', 'O2', 'Pz', 'O1', 'HEOG', 'x_dir', 'y_dir', 'z_dir')
electrodes_used_names = ('P3', 'P4', 'Pz', 'O1', 'O2')


plt.figure()
plt.title('All Phase shifts')

small_dot_size = 8

colours = ['r','m','g','b','c','y','k']

electrode_count = 0

for electrode in electrodes_to_use:
    
    for frequency_count in range(0,6):

        # difference
        phase_scores_all_subjects =  SSVEP_phase_scores[:,frequency_count,electrode]

      #  plt.scatter(np.zeros([10,])+((electrode_count*10) + frequency_count),phase_scores_all_subjects, c=colours[frequency_count], s=small_dot_size)

        mean_phase_difference = phase_scores_all_subjects.mean()
        
        std_error_phase_difference = np.std(phase_scores_all_subjects) / math.sqrt(10)

        plt.errorbar((electrode_count*10) + frequency_count, mean_phase_difference,yerr = std_error_phase_difference, solid_capstyle='projecting', capsize=5,  fmt='o', color= colours[frequency_count], ecolor=colours[frequency_count])  

    if electrode == 5: # electrode Pz is the only electrode where the signal generator is approximatly the same size as real SSVEPs
        ## SIGI
    
        SIGI_phase_difference_all_subjects = SIGI_phase_scores[:,frequency_count,electrode]
      
       # plt.scatter(np.zeros([10,])+((electrode_count*10) + 6),SIGI_phase_difference_all_subjects, c=colours[6], s=small_dot_size)
    
        mean_SIGI_phase_difference = SIGI_phase_difference_all_subjects.mean()
            
        std_error_phase_SIGI_difference = np.std(SIGI_phase_difference_all_subjects) / math.sqrt(10)
            
        plt.errorbar((electrode_count*10) + 6, mean_SIGI_phase_difference,yerr = std_error_phase_SIGI_difference, solid_capstyle='projecting', capsize=5,  fmt='o', color= colours[6], ecolor=colours[6])  
    
    electrode_count += 1

    
# zero_line = np.arange(0,80)
# plt.plot(zero_line,np.zeros([80,]),'k--')

plt.ylabel('Absolute Phase Shift (degrees)')

plt.errorbar(-100, 0,yerr = 0, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'r', ecolor='r', label = ' 30 Hz') 
plt.errorbar(-100, 0,yerr = 0, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'm', ecolor='m', label = ' 35 Hz')  
plt.errorbar(-100, 0,yerr = 0, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'g', ecolor='g', label = ' 40 Hz') 
plt.errorbar(-100, 0,yerr = 0, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'b', ecolor='b', label = ' 45 Hz')  
plt.errorbar(-100, 0,yerr = 0, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'c', ecolor='c', label = ' 50 Hz') 
plt.errorbar(-100, 0,yerr = 0, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'y', ecolor='y', label = ' 55 Hz')  
plt.errorbar(-100, 0,yerr = 0, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'k', ecolor='k', label = ' Signal Generator \n (40 Hz)') 

plt.legend()

## set x axis ticks
x = np.arange(0,50,10)+3
y = np.zeros([5,])
labels = electrodes_used_names
plt.xticks(x, labels, rotation='vertical')
plt.show()

plt.xlim(-1, 48)




## plot phase shifts: just P3, Pz, P4
electrode = 5

frequency_names = ('30 Hz', '35 Hz', '40 Hz', '45 Hz', '50 Hz', '55 Hz', 'Signal \n Generator \n (40 Hz)')
small_dot_size = 8

colours = ['r','m','g','b','c','y','k']


plt.figure()
plt.suptitle('Phase shifts')

plot_count = 1

for electrode in (2, 5, 3):
    
    plt.subplot(1,3,plot_count)
    if plot_count == 1:
        plt.ylabel('Absolute Phase Shift (degrees)')
    
    plot_count +=1

    plt.title(electrode_names[electrode])
    
    
    
    for frequency_count in range(0,6):
    
        # phase shift
        phase_scores_all_subjects =  SSVEP_phase_scores[:,frequency_count,electrode]
    
       # plt.scatter(np.zeros([10,]) + frequency_count,phase_scores_all_subjects, c=colours[frequency_count], s=small_dot_size)
    
        mean_phase_difference = phase_scores_all_subjects.mean()
        
        std_error_phase_difference = np.std(phase_scores_all_subjects) / math.sqrt(10)
    
        plt.errorbar(frequency_count, mean_phase_difference,yerr = std_error_phase_difference, solid_capstyle='projecting', capsize=5,  fmt='o', color= colours[frequency_count], ecolor=colours[frequency_count])  
    
    
    ## SIGI
    
    SIGI_phase_difference_all_subjects = SIGI_phase_scores[:,frequency_count,electrode]
      
   # plt.scatter(np.zeros([10,]) + 6,SIGI_phase_difference_all_subjects, c=colours[6], s=small_dot_size)
    
    mean_SIGI_phase_difference = SIGI_phase_difference_all_subjects.mean()
        
    std_error_phase_SIGI_difference = np.std(SIGI_phase_difference_all_subjects) / math.sqrt(10)
        
    plt.errorbar(6, mean_SIGI_phase_difference,yerr = std_error_phase_SIGI_difference, solid_capstyle='projecting', capsize=5,  fmt='o', color= colours[6], ecolor=colours[6])  
    
        
    # zero_line = np.arange(0,80)
    # plt.plot(zero_line,np.zeros([80,]),'k--')
    
    plt.ylim(-5, 105)
    plt.xlim(-0.5, 6.5)
    
    ## set x axis ticks
    ## set x axis ticks
    x = np.arange(0,7,1)
    y = np.zeros([7,])
    labels = frequency_names
    plt.xticks(x, labels, rotation='vertical')
    plt.show()


plt.errorbar(-100, 0,yerr = 0, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'r', ecolor='r', label = ' 30 Hz') 
plt.errorbar(-100, 0,yerr = 0, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'm', ecolor='m', label = ' 35 Hz')  
plt.errorbar(-100, 0,yerr = 0, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'g', ecolor='g', label = ' 40 Hz') 
plt.errorbar(-100, 0,yerr = 0, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'b', ecolor='b', label = ' 45 Hz')  
plt.errorbar(-100, 0,yerr = 0, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'c', ecolor='c', label = ' 50 Hz') 
plt.errorbar(-100, 0,yerr = 0, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'y', ecolor='y', label = ' 55 Hz')  
plt.errorbar(-100, 0,yerr = 0, solid_capstyle='projecting', capsize=5,  fmt='o', color= 'k', ecolor='k', label = ' Signal Generator \n (40 Hz)') 
plt.legend()

plt.xlim(-0.5, 6.5)





#### all correlations


# plt.figure()
# plt.suptitle('All correlations')

# small_dot_size = 2

# colours = ['r','m','g','b','c','y','k']

# for electrode in range(0,8):
    
#     for frequency_count in range(0,6):

#         # difference
#         correlations_all_subjects =  SSVEP_walking_standing_correlations[:,frequency_count,electrode]

#         plt.scatter(np.zeros([10,])+((electrode*10) + frequency_count),correlations_all_subjects, c=colours[frequency_count], s=small_dot_size)

#         mean_correlation = correlations_all_subjects.mean()
        
#         std_error_correlations = np.std(correlations_all_subjects) / math.sqrt(10)

#         plt.errorbar((electrode*10) + frequency_count, mean_correlation,yerr = std_error_correlations, solid_capstyle='projecting', capsize=5,  fmt='o', color= colours[frequency_count], ecolor=colours[frequency_count])  


#     if electrode == 5: # electrode Pz is the only electrode where the signal generator is approximatly the same size as real SSVEPs
#     ## SIGI
    
#         SIGI_correlations_all_subjects = SIGI_walking_standing_correlations[:,frequency_count,electrode]
      
#         plt.scatter(np.zeros([10,])+((electrode*10) + 6),SIGI_correlations_all_subjects, c=colours[6], s=small_dot_size)
    
#         mean_SIGI_correlation = SIGI_correlations_all_subjects.mean()
            
#         std_error_SIGI_correlations = np.std(SIGI_correlations_all_subjects) / math.sqrt(10)
            
#         plt.errorbar((electrode*10) + 6, mean_SIGI_correlation,yerr = std_error_SIGI_correlations, solid_capstyle='projecting', capsize=5,  fmt='o', color= colours[6], ecolor=colours[6])  

    
# # zero_line = np.arange(0,80)
# # plt.plot(zero_line,np.zeros([80,]),'k--')

# ## set x axis ticks
# x = np.arange(0,80,10)+3
# y = np.zeros([8,])
# labels = electrode_names[0:8]
# plt.xticks(x, labels, rotation='vertical')
# plt.show()




########### Stats  ####################

## Amplitude compared to Blackout condition

Z_scores_amplitude_compared_to_blackout = np.zeros([8,6])
p_values_amplitude_compared_to_blackout = np.zeros([8,6])
effect_size_amplitude_compared_to_blackout = np.zeros([8,6])

for electrode in range(0,8):  #'VEOG', 'blink', 'P3', 'P4', 'O2', 'Pz', 'O1', 'HEOG'
    
    all_subjects_blackout =  blackout_amplitudes[:,electrode] 

    print(' ')
    print('Electrode ' +  electrode_names[electrode])
    print(' ')

    for frequency_count in range(0,6):
    
       # standing
        print(' ')
        print('Standing ' + str(frequencies_to_use[frequency_count]) + ' Hz')

        all_subjects_amplitudes_standing = SSVEP_amplitudes[:,frequency_count,electrode,0]

        Z_score = functions.group_permutation_test(all_subjects_amplitudes_standing, all_subjects_blackout)

        p_value_one_sided = scipy.stats.norm.sf(abs(Z_score)) #one-sided

        p_value_two_sided = scipy.stats.norm.sf(abs(Z_score))*2 #twosided

        cohens_d = functions.cohens_d(all_subjects_amplitudes_standing, all_subjects_blackout)
        
        print('p = ' + str(p_value_two_sided))
        print('cohens d = ' + str(cohens_d))
        print('  ')
        
        
        # walking
        print('Walking ' + str(frequencies_to_use[frequency_count]) + ' Hz')
        
        # standing
        all_subjects_amplitudes_walking = SSVEP_amplitudes[:,frequency_count,electrode,1]

        Z_score = functions.group_permutation_test(all_subjects_amplitudes_walking, all_subjects_blackout)

        p_value_one_sided = scipy.stats.norm.sf(abs(Z_score)) #one-sided

        p_value_two_sided = scipy.stats.norm.sf(abs(Z_score))*2 #twosided

        cohens_d = functions.cohens_d(all_subjects_amplitudes_walking, all_subjects_blackout)
        
        print('p = ' + str(p_value_one_sided))
        print('cohens d = ' + str(cohens_d))
        print('  ')
        
        Z_scores_amplitude_compared_to_blackout[electrode,frequency_count] = Z_score
        p_values_amplitude_compared_to_blackout[electrode,frequency_count] = p_value_one_sided
        effect_size_amplitude_compared_to_blackout[electrode,frequency_count] = cohens_d
        
        
        
# read out average effect sizes
print('  ')
print('Average effect size of amplitude vs. blackout (Cohen''s d) across electrodes: ')
print(' ')
frequency_names = ('30 Hz', '35 Hz', '40 Hz', '45 Hz', '50 Hz', '55 Hz', 'Signal \n Generator \n (40 Hz)')    
electrodes_to_use = (2,3,5,6,4) #  ('VEOG', 'blink', 'P3', 'P4', 'O2', 'Pz', 'O1', 'HEOG', 'x_dir', 'y_dir', 'z_dir')
electrodes_used_names = ('P3', 'P4', 'Pz', 'O1', 'O2')

for frequency_count in range(0,6):
    
    average_effect_size_for_frequency = effect_size_amplitude_compared_to_blackout[electrodes_to_use,frequency_count].mean()
    print(str(frequency_names[frequency_count]) + ' average effect size = ' + str(average_effect_size_for_frequency))


    
        
#### compare phase shift to Pz SIGI condition

Z_scores_phase_shift_compared_to_Pz_SIGI = np.zeros([8,6])
p_values_phase_shift_compared_to_Pz_SIGI = np.zeros([8,6])
effect_size_phase_shift_compared_to_Pz_SIGI = np.zeros([8,6])


      
for electrode in range(0,8):  #'VEOG', 'blink', 'P3', 'P4', 'O2', 'Pz', 'O1', 'HEOG'
    
    print(' ')
    print('Electrode ' +  electrode_names[electrode])
    print(' ')
      
    for frequency_count in range(0,6):
        
        print(' ')
        print(str(frequencies_to_use[frequency_count]) + ' Hz')
        
        all_Pz_SIGI_Phase_shifts = SIGI_phase_scores[:,frequency_count,5]
       
        all_phase_shifts = SSVEP_phase_scores[:,frequency_count,electrode]
        
       
        Z_score = functions.group_permutation_test(all_phase_shifts, all_Pz_SIGI_Phase_shifts)

        p_value_one_sided = scipy.stats.norm.sf(abs(Z_score)) #one-sided

        cohens_d = functions.cohens_d(all_phase_shifts, all_Pz_SIGI_Phase_shifts)
        
        print('p = ' + str(p_value_one_sided))
        print('cohens d = ' + str(cohens_d))
        print('  ')        
       
        Z_scores_phase_shift_compared_to_Pz_SIGI[electrode,frequency_count] = Z_score
        p_values_phase_shift_compared_to_Pz_SIGI[electrode,frequency_count] = p_value_one_sided
        effect_size_phase_shift_compared_to_Pz_SIGI[electrode,frequency_count] = cohens_d
       
# read out average effect sizes
print('  ')
print('Average effect size of phase shift vs. Signal generator (Cohen''s d) across electrodes: ')
print(' ')
frequency_names = ('30 Hz', '35 Hz', '40 Hz', '45 Hz', '50 Hz', '55 Hz', 'Signal \n Generator \n (40 Hz)')    
electrodes_to_use = (2,3,5,6,4) #  ('VEOG', 'blink', 'P3', 'P4', 'O2', 'Pz', 'O1', 'HEOG', 'x_dir', 'y_dir', 'z_dir')
electrodes_used_names = ('P3', 'P4', 'Pz', 'O1', 'O2')

for frequency_count in range(0,6):
    
    average_effect_size_for_frequency = effect_size_phase_shift_compared_to_Pz_SIGI[electrodes_to_use,frequency_count].mean()
    print(str(frequency_names[frequency_count]) + ' average effect size = ' + str(average_effect_size_for_frequency))

       
       
        
### compare amplitude difference to Pz SIGI condition


for electrode in range(0,8):  #'VEOG', 'blink', 'P3', 'P4', 'O2', 'Pz', 'O1', 'HEOG'
    
    print(' ')
    print('Electrode ' +  electrode_names[electrode])
    print(' ')
      
    for frequency_count in range(0,6):
        
        print(' ')
        print(str(frequencies_to_use[frequency_count]) + ' Hz')
        
        all_Pz_SIGI_amplitude_differences = SIGI_amplitudes[:,frequency_count,5,1] - SIGI_amplitudes[:,frequency_count,5,0] # walking minus standing
       
        all_amplitude_differences = SSVEP_amplitudes[:,frequency_count,electrode,1] - SSVEP_amplitudes[:,frequency_count,electrode,0] # walking minus standing
        
        Z_score = functions.group_permutation_test(all_amplitude_differences, all_Pz_SIGI_amplitude_differences)

        p_value_one_sided = scipy.stats.norm.sf(abs(Z_score)) #one-sided
        
        p_value_two_sided = scipy.stats.norm.sf(abs(Z_score))*2 #twosided

        cohens_d = functions.cohens_d(all_amplitude_differences, all_Pz_SIGI_amplitude_differences)
        
        print('p = ' + str(p_value_two_sided))
        print('cohens d = ' + str(cohens_d))
        print('  ')        
       
        




