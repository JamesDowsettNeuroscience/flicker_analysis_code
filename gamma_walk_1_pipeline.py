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

import os

### information about the experiment: Gamma walk 1

#path = '/home/james/Active_projects/Gamma_walk/Gamma_walking_experiment_1/raw_data_for_analysis_package/'

path = 'D:\\Gamma_walk\\Gamma_walking_experiment_1\\raw_data_for_analysis_package'



electrode_names = ('VEOG', 'blink', 'P3', 'P4', 'O2', 'Pz', 'O1', 'HEOG', 'x_dir', 'y_dir', 'z_dir')

condition_names = ('standing', 'walking')

sample_rate = 1000

num_subjects = 10

frequencies_to_use = (30, 35, 40, 45, 50, 55)


trig_1_times = [-1, -1, -1, -1, -1, -1]
trig_2_times = [15, 13, 11, 10, 9, 8]
trig_length = 4

#######################################


# ### Matrices to store results and SSVEPs

# SIGI_SSVEPs = np.zeros([num_subjects,6,8,2,25]) # subject, frequency, electrode, condition, SSVEP data (SIGI always 40 Hz, lenght = 25)

# SIGI_amplitudes = np.zeros([num_subjects,6,8,2]) # subject, frequency, electrode, condition

# SIGI_walking_standing_correlations = np.zeros([num_subjects,6,8])

# SIGI_phase_scores = np.zeros([num_subjects,6,8])


# all_SSVEPs = np.zeros([num_subjects,6,8,2,34]) # subject, frequency, electrode, condition, SSVEP data (34 data points is the largest SSVEP)

# SSVEP_amplitudes = np.zeros([num_subjects,6,8,2]) # subject, frequency, electrode , condition

# SSVEP_walking_standing_correlations = np.zeros([num_subjects,6,8]) # subject, frequency, electrode

# SSVEP_phase_scores = np.zeros([num_subjects,6,8])



# blackout_SSVEPs = np.zeros([num_subjects,8,25]) # blackout was 40 Hz, so length = 25

# blackout_amplitudes = np.zeros([num_subjects,8])



# all_mean_self_absolute_phase_shifts = np.zeros([num_subjects,6,8,2])  # subject, frequency, electrode, condition
# self_split_amplitude_differences = np.zeros([num_subjects,6,8,2])  # subject, frequency, electrode, condition

# all_mean_self_split_correlations = np.zeros([num_subjects,6,8,2])  # subject, frequency, electrode, condition


# length = 1 # length of FFT in seconds
# all_evoked_FFTs = np.zeros([num_subjects,6,8,2,(length * sample_rate)]) # subject, frequency, electrode, condition, FFT data 


# ##################################

# for subject in range(1,11):
   
#     print('  ')
#     print('Subject ' + str(subject))
#     print(' ')
 
    
#     for electrode in range(0,8):
        
#         electrode_name = electrode_names[electrode]
        
#         print(' ')
#         print(electrode_name)
#         print(' ')
        
#         ## load raw data
        
#         data_file_name = 'subject_' + str(subject) + '_electrode_' + str(electrode) + '_data.npy'
        
#         data_with_path = os.path.join(path,data_file_name)
        
#         raw_data = np.load(data_with_path)                



#         ####### SIGI conditions
        
#         frequency_count = 0
#         for frequency in frequencies_to_use: # loop for each frequency to match the number of segments from each frequency 

#             for condition in range(0,2):
                
#                 ## load triggers from real SSVEP condition to match the number of triggers to use
#                 triggers_file_name = 'subject_' + str(subject) + '_' + condition_names[condition] + '_' + str(frequency) + 'Hz_triggers.npy'   
                
#                 triggers_with_path = os.path.join(path,triggers_file_name)
                
#                 triggers = np.load(triggers_with_path)    
            
#                 num_triggers_to_use = len(triggers)
                
                
#                 # load the SIGI triggers
#                 triggers_file_name = 'subject_' + str(subject) + '_SIGI_' + condition_names[condition] + '_triggers.npy'
                
#                 triggers_with_path = os.path.join(path,triggers_file_name)
                
#                 triggers = np.load(triggers_with_path)    
                
#                 # only use the same number of triggers that there were in the real SSVEP condition
#                 triggers = triggers[0:num_triggers_to_use]
                
#                 print(condition_names[condition] + ' ' + str(len(triggers)))
                
#                 ### make SSVEP
                
#                 period = int(np.round(sample_rate/40))
                
#                 SSVEP = functions.make_SSVEPs(raw_data, triggers, 25) # SIGI was always 40 Hz, length = 25

#                 # plt.plot(SSVEP)
        
#                 SIGI_amplitudes[subject-1,frequency_count,electrode,condition] = np.ptp(SSVEP) # save amplitude
                
#                 SIGI_SSVEPs[subject-1,frequency_count,electrode,condition,:] = SSVEP # save the SSVEP
                

#                     # make a copy to later compare walking and standing
#                 if condition == 0:
#                     standing_SSVEP = np.copy(SSVEP)
#                 elif condition== 1:
#                     walking_SSVEP = np.copy(SSVEP)
        
        
#             # get walking/standing correlations and phase shift
#             SIGI_walking_standing_correlations[subject-1,frequency_count,electrode] = np.corrcoef(standing_SSVEP, walking_SSVEP)[0,1]
        
#             SIGI_phase_scores[subject-1,frequency_count,electrode] = functions.cross_correlation_absolute(standing_SSVEP, walking_SSVEP)
        
#             frequency_count += 1
            
            
    
#         ######### make real SSVEPs and FFTs ########################
#         frequency_count = 0
#         for frequency in frequencies_to_use:
#             for condition in range(0,2):
            
            
#                 ## load triggers
#                 triggers_file_name = 'subject_' + str(subject) + '_' + condition_names[condition] + '_' + str(frequency) + 'Hz_triggers.npy'
                
#                 triggers_with_path = os.path.join(path,triggers_file_name)
                
#                 triggers = np.load(triggers_with_path)
                
#                 ### linear interpolation
#                 data_linear_interpolation = functions.linear_interpolation(raw_data, triggers, trig_1_times[frequency_count], trig_2_times[frequency_count], trig_length)
                
                
                
#                 ### make SSVEP
                
#                 period = int(np.round(sample_rate/frequency))
                
#                 SSVEP = functions.make_SSVEPs(data_linear_interpolation, triggers, period)

#                 # save amplitude
#                 SSVEP_amplitudes[subject-1,frequency_count,electrode,condition] = np.ptp(SSVEP)
                
#                 # save the SSVEP
#                 all_SSVEPs[subject-1,frequency_count,electrode,condition,0:len(SSVEP)] = SSVEP 
                
#                 # get absolute self phase shift
#                 phase_shift = functions.phase_shift_SSVEPs_split(data_linear_interpolation, triggers, period)
                
#                 all_mean_self_absolute_phase_shifts[subject-1, frequency_count, electrode, condition] = np.abs(phase_shift)
                
#                 ## get self correlation
                
#                 self_split_correlation = functions.compare_SSVEPs_split(data_linear_interpolation, triggers, period)
#                 all_mean_self_split_correlations[subject-1, frequency_count, electrode, condition] = self_split_correlation
                
#                 # get self amplitude difference
#                 amplitude_difference = functions.SSVEP_split_amplitude_difference(data_linear_interpolation, triggers, period)
#                 self_split_amplitude_differences[subject-1, frequency_count, electrode, condition] = amplitude_difference
                
#                 # Evoked FFT
#                 # evoked_FFT = functions.evoked_fft(data_linear_interpolation, triggers, length, sample_rate)
#                 # all_evoked_FFTs[subject-1,frequency_count,electrode,condition,0:len(evoked_FFT)] = evoked_FFT # subject, frequency, electrode, condition, FFT data 
                
#                 # make a copy to later compare walking and standing
#                 if condition == 0:
#                     standing_SSVEP = np.copy(SSVEP)
#                 elif condition== 1:
#                     walking_SSVEP = np.copy(SSVEP)
                    
#             # save correlations and phase shift
#             SSVEP_walking_standing_correlations[subject-1,frequency_count,electrode] = np.corrcoef(standing_SSVEP, walking_SSVEP)[0,1]
               
#             SSVEP_phase_scores[subject-1,frequency_count,electrode] = functions.cross_correlation_absolute(standing_SSVEP, walking_SSVEP)

#             frequency_count += 1
                    
                
#     ############# make blackout SSVEPs  ######################
    
   
        
#         ## load triggers
#         triggers_file_name = 'subject_' + str(subject) + '_blackout_triggers.npy'
        
#         triggers_with_path = os.path.join(path,triggers_file_name)
        
#         triggers = np.load(triggers_with_path)
        
#         ### linear interpolation, use 40 Hz trigger times, = frequency 2
#         data_linear_interpolation = functions.linear_interpolation(raw_data, triggers, trig_1_times[2], trig_2_times[2], trig_length)
        
         
#         period = int(np.round(sample_rate/40))
        
#         SSVEP = functions.make_SSVEPs(data_linear_interpolation, triggers, period)
        
#         blackout_amplitudes[subject-1,electrode] = np.ptp(SSVEP)

#         blackout_SSVEPs[subject-1,electrode,:] = SSVEP # save the SSVEP




### save all the SSVEPs and FFT spectrums, because this takes some time

#np.save(path + 'all_evoked_FFTs', all_evoked_FFTs)

# np.save(os.path.join(path,'SIGI_SSVEPs'),SIGI_SSVEPs)

# np.save(os.path.join(path,'SIGI_amplitudes'),SIGI_amplitudes)

# np.save(os.path.join(path,'SIGI_walking_standing_correlations'),SIGI_walking_standing_correlations)
 
# np.save(os.path.join(path,'SIGI_phase_scores'),SIGI_phase_scores)
 

# np.save(os.path.join(path,'all_SSVEPs'),all_SSVEPs)

# np.save(os.path.join(path,'SSVEP_amplitudes'),SSVEP_amplitudes)
 
# np.save(os.path.join(path,'SSVEP_walking_standing_correlations'),SSVEP_walking_standing_correlations)

# np.save(os.path.join(path,'SSVEP_phase_scores'),SSVEP_phase_scores)



# np.save(os.path.join(path,'blackout_SSVEPs'),blackout_SSVEPs)

# np.save(os.path.join(path,'blackout_amplitudes'),blackout_amplitudes)
 

 
# np.save(os.path.join(path,'all_mean_self_absolute_phase_shifts'),all_mean_self_absolute_phase_shifts)

# np.save(os.path.join(path,'self_split_amplitude_differences'),self_split_amplitude_differences)

# np.save(os.path.join(path,'all_mean_self_split_correlations'),all_mean_self_split_correlations)




## load all the above data

all_evoked_FFTs = np.load(os.path.join(path, 'all_evoked_FFTs.npy'))


SIGI_SSVEPs = np.load(os.path.join(path, 'SIGI_SSVEPs.npy'))

SIGI_amplitudes =  np.load(os.path.join(path, 'SIGI_amplitudes.npy'))

SIGI_walking_standing_correlations = np.load(os.path.join(path, 'SIGI_walking_standing_correlations.npy'))

SIGI_phase_scores = np.load(os.path.join(path, 'SIGI_phase_scores.npy'))


all_SSVEPs = np.load(os.path.join(path, 'all_SSVEPs.npy'))

SSVEP_amplitudes = np.load(os.path.join(path, 'SSVEP_amplitudes.npy'))

SSVEP_walking_standing_correlations = np.load(os.path.join(path, 'SSVEP_walking_standing_correlations.npy')) 

SSVEP_phase_scores = np.load(os.path.join(path, 'SSVEP_phase_scores.npy'))



blackout_SSVEPs = np.load(os.path.join(path, 'blackout_SSVEPs.npy'))


blackout_amplitudes = np.load(os.path.join(path, 'blackout_amplitudes.npy'))



all_mean_self_absolute_phase_shifts = np.load(os.path.join(path, 'all_mean_self_absolute_phase_shifts.npy'))

self_split_amplitude_differences = np.load(os.path.join(path, 'self_split_amplitude_differences.npy'))

all_mean_self_split_correlations = np.load(os.path.join(path, 'all_mean_self_split_correlations.npy'))




# ## Evoked FFT spectrums

#electrode = 5

peak_locations = (31, 36, 42, 45, 50, 55)
first_harmonic_locations = (62, 71, 84, 90, 100, 110)

peak_amplitudes = np.zeros([10,6,8,2])
first_harmonic_amplitudes = np.zeros([10,6,8,2])

# empty matrix to store the noise values, i.e. the average of the FFT -5 to 5 Hz around the peak
average_noise_peak = np.zeros([10,6,8,2])
average_noise_first_harmonic = np.zeros([10,6,8,2])

SNR_peaks = np.zeros([10,6,8,2])
SNR_first_harmonic = np.zeros([10,6,8,2])

for subject in range(0,10):
    # plt.figure()
    # plt.suptitle('Subject ' + str(subject+1))
    for frequency in range(0,6):
        # plt.subplot(3,2,frequency+1)
        # plt.title(str(frequencies_to_use[frequency]) + ' Hz')
        
        for electrode in range(0,8):
            for condition in range(0,2):
                
                fft_spectrum = all_evoked_FFTs[subject,frequency,electrode,condition,:]
                
              #  plt.plot(fft_spectrum, label = (str(condition_names[condition])))
                
                peak_frequency = peak_locations[frequency]
     
                peak_amplitude = fft_spectrum[peak_frequency] 
                peak_noise_amplitude = fft_spectrum[np.r_[peak_frequency-5:peak_frequency-2, peak_frequency+2:peak_frequency+5]]
                
                peak_amplitudes[subject,frequency,electrode,condition] = peak_amplitude
                average_noise_peak[subject,frequency,electrode,condition] = peak_noise_amplitude.mean()
                SNR_peaks[subject,frequency,electrode,condition] = peak_amplitude / peak_noise_amplitude.mean()
                
                first_harmonic_frequency = first_harmonic_locations[frequency]
                
                first_harmonic_amplitude = fft_spectrum[first_harmonic_frequency] 
                
                first_harmonic_noise_amplitude = fft_spectrum[np.r_[first_harmonic_frequency-5:first_harmonic_frequency-2, first_harmonic_frequency+2:first_harmonic_frequency+5]]
                
                first_harmonic_amplitudes[subject,frequency,electrode,condition] = first_harmonic_amplitude 
                average_noise_first_harmonic[subject,frequency,electrode,condition] = first_harmonic_noise_amplitude.mean()
                SNR_first_harmonic[subject,frequency,electrode,condition] = first_harmonic_amplitude / first_harmonic_noise_amplitude.mean()
                
               
       # plt.xlim(0, 100)
    
   # plt.legend()





## plot grand average FFTs
for electrode in range(0,8):
    plt.figure()
    plt.suptitle(electrode_names[electrode])
    print('\n ' + str(electrode_names[electrode]) +  '\n')
    for frequency in range(0,6):
        plt.subplot(3,2,frequency+1)
        plt.title(str(frequencies_to_use[frequency]) + ' Hz')
        for condition in range(0,2):
            
            grand_average_fft = all_evoked_FFTs[:,frequency,electrode,condition,:].mean(axis=0)
            plt.plot(grand_average_fft, label = (str(condition_names[condition])))
            plt.xlim(20, 120)
            plt.ylim(0, 135)
            
            peak_amplitudes_all_subjects = peak_amplitudes[:, frequency, electrode, condition]
            first_harmonic_all_subjects = first_harmonic_amplitudes[:, frequency, electrode, condition]
            
            SNR_peaks_all_subjects = SNR_peaks[:, frequency, electrode, condition]
            SNR_first_harmonic_all_subjects = SNR_first_harmonic[:, frequency, electrode, condition]
            
            average_peak_SNR = SNR_peaks_all_subjects.mean()
            average_first_harmonic_SNR = SNR_first_harmonic_all_subjects.mean()
            
            # harmonic_ratios = peak_amplitudes_all_subjects / first_harmonic_all_subjects
            
            # average_harmonic_ratio = harmonic_ratios.mean()
            
            print(str(frequencies_to_use[frequency]) + ' Hz  '  + str(condition_names[condition]) + '  peak = ' + str(average_peak_SNR) + '  , first harmonic =  ' + str(average_first_harmonic_SNR))
            
    plt.legend()        




######  plots

## check raw SSVEPs for each electrode

# electrode = 5 #('VEOG', 'blink', 'P3', 'P4', 'O2', 'Pz', 'O1', 'HEOG', 'x_dir', 'y_dir', 'z_dir')

# for subject in range(1,11):
    
#     plt.figure()
#     plt.suptitle('Subject ' + str(subject) + ' ' + electrode_names[electrode])
    
#     for frequency_count in range(0,6):
        
#         plt.subplot(3,3,frequency_count+1)
        
#         plt.title(str(frequencies_to_use[frequency_count]) + ' Hz')
        
#         period = int(np.round(sample_rate/frequencies_to_use[frequency_count]))
        
#         standing_SSVEP = all_SSVEPs[subject-1,frequency_count,electrode,0,0:period]
#         walking_SSVEP = all_SSVEPs[subject-1,frequency_count,electrode,1,0:period]
        
#         plt.plot(standing_SSVEP,'b')
#         plt.plot(walking_SSVEP,'r')

    
#     plt.subplot(3,3,3)

#     blackout_SSVEP =  blackout_SSVEPs[subject-1,electrode,:] 
    
#     plt.plot(blackout_SSVEP,'k')


#     plt.subplot(3,3,8)
#     for frequency_count in range(0,6):
        
#         standing_SIGI = SIGI_SSVEPs[subject-1,frequency_count,electrode,0,:]
#         walking_SIGI = SIGI_SSVEPs[subject-1,frequency_count,electrode,1,:]

#         plt.plot(standing_SIGI,'b')
#         plt.plot(walking_SIGI,'r')



## plot raw SSVEPs for a single electrode and frequency

electrode = 5 #('VEOG', 'blink', 'P3', 'P4', 'O2', 'Pz', 'O1', 'HEOG', 'x_dir', 'y_dir', 'z_dir')

frequency_count = 2

plt.figure()
plt.suptitle(electrode_names[electrode] + '  ' + str(frequencies_to_use[frequency_count]) + ' Hz')
    
for subject in range(1,11):
    

    plt.subplot(3,4,subject)
    
    plt.title(str(subject))
    
    period = int(np.round(sample_rate/frequencies_to_use[frequency_count]))
    
    standing_SSVEP = all_SSVEPs[subject-1,frequency_count,electrode,0,0:period]
    walking_SSVEP = all_SSVEPs[subject-1,frequency_count,electrode,1,0:period]
    
    plt.plot(standing_SSVEP,'b', label = 'Standing')
    plt.plot(walking_SSVEP,'r', label = 'Walking')


    # blackout_SSVEP =  blackout_SSVEPs[subject-1,electrode,:] 
    
    # plt.plot(blackout_SSVEP,'k', label = 'Blackout')

   # plt.ylim(-1, 1)

plt.legend()




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

electrodes_to_use =  (2, 5, 3)  #(2, 3)  #

for electrode in electrodes_to_use:
    
    plt.subplot(1,len(electrodes_to_use),plot_count)
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




#### plot P3 P4 only: 35 and 40 Hz differences


plt.figure()

x_position = 0

P3_P4_phase_scores = np.zeros([10,2,2])

for frequency_count in (1,2):
    
    average_phase_shifts = np.zeros([2,])
    
    electrode_count = 0
    
    for electrode in (2,3):

        # phase shift
        phase_scores_all_subjects =  SSVEP_phase_scores[:,frequency_count,electrode]
    
       # plt.scatter(np.zeros([10,]) + frequency_count,phase_scores_all_subjects, c=colours[frequency_count], s=small_dot_size)
    
        mean_phase_difference = phase_scores_all_subjects.mean()
        
        average_phase_shifts[electrode_count] = mean_phase_difference
        
        std_error_phase_difference = np.std(phase_scores_all_subjects) / math.sqrt(10)
    
        plt.errorbar(x_position, mean_phase_difference,yerr = std_error_phase_difference, solid_capstyle='projecting', capsize=5,  fmt='o', color= colours[frequency_count], ecolor=colours[frequency_count])  


        P3_P4_phase_scores[:,frequency_count-1,electrode_count] = phase_scores_all_subjects

        x_position = x_position + 1   
        electrode_count += 1

    if frequency_count == 1:
        plt.plot((0,1),average_phase_shifts, color= colours[frequency_count],  label = frequency_names[frequency_count]) 
    elif frequency_count == 2:
        plt.plot((2,3),average_phase_shifts, color= colours[frequency_count],  label = frequency_names[frequency_count]) 

    


plt.ylabel('Absolute Phase Shift (degrees)')

plt.xticks((0,1,2,3), ('P3', 'P4','P3', 'P4'))

plt.title('Walking-Standing phase shift')

plt.legend()


## stats

phase_P3_P4_difference_35Hz = P3_P4_phase_scores[:,0,0] -  P3_P4_phase_scores[:,0,1]
phase_P3_P4_difference_40Hz = P3_P4_phase_scores[:,1,0] -  P3_P4_phase_scores[:,1,1]

Z_score_35_vs_40_Hz = functions.group_permutation_test(phase_P3_P4_difference_35Hz, phase_P3_P4_difference_40Hz)



Z_score_phase_P3_vs_P4_35Hz = functions.group_permutation_test(P3_P4_phase_scores[:,0,0],  P3_P4_phase_scores[:,0,1])

Z_score_phase_P3_vs_P4_40Hz = functions.group_permutation_test(P3_P4_phase_scores[:,1,0],  P3_P4_phase_scores[:,1,1])

Z_score_phase_35_vs_40z_P3 = functions.group_permutation_test(P3_P4_phase_scores[:,0,0],  P3_P4_phase_scores[:,1,0])

Z_score_phase_35_vs_40z_P4 = functions.group_permutation_test(P3_P4_phase_scores[:,0,1],  P3_P4_phase_scores[:,1,1])


print('\nPhase shifts:')
print('P3 vs P4, 35Hz Z = ' + str(Z_score_phase_P3_vs_P4_35Hz))
print('P3 vs P4, 40Hz Z = ' + str(Z_score_phase_P3_vs_P4_40Hz))
print('35 vs 40 Hz, P3 Z = ' + str(Z_score_phase_35_vs_40z_P3))
print('35 vs 40 Hz, P4 Z = ' + str(Z_score_phase_35_vs_40z_P4))




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


## Evoked FFTs, check for presence of 2nd harmonic

electrode = 5

for frequency_count in range(0,6):
    
    print('\n' + str(frequencies_to_use[frequency_count]) + ' Hz\n')
    
    for condition in range(0,2):
        
        print(condition_names[condition] )
    
        first_harmonic_amplitudes_all_subjects = first_harmonic_amplitudes[:,frequency_count,electrode,condition]
        
        average_noise_first_harmonic_all_subjects = average_noise_first_harmonic[:,frequency_count,electrode,condition]

        Z_score = functions.group_permutation_test(first_harmonic_amplitudes_all_subjects ,average_noise_first_harmonic_all_subjects)

        print(str(Z_score))
        
        p_value_one_sided = scipy.stats.norm.sf(abs(Z_score)) #one-sided
        
        print(str(p_value_one_sided))

        cohens_d = functions.cohens_d(first_harmonic_amplitudes_all_subjects, average_noise_first_harmonic_all_subjects)

        print('cohens d = ' + str(cohens_d) + '\n')







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

frequency_names = ('30 Hz', '35 Hz', '40 Hz', '45 Hz', '50 Hz', '55 Hz', 'Signal \n Generator \n (40 Hz)')    
electrodes_to_use = (2,3,5,6,4) #  ('VEOG', 'blink', 'P3', 'P4', 'O2', 'Pz', 'O1', 'HEOG', 'x_dir', 'y_dir', 'z_dir')
electrodes_used_names = ('P3', 'P4', 'Pz', 'O1', 'O2')

print('  ')
print('Average effect size of amplitude vs. blackout (Cohen''s d) across electrodes: ')
print(' ')
for frequency_count in range(0,6):
    average_effect_size_for_frequency = effect_size_amplitude_compared_to_blackout[electrodes_to_use,frequency_count].mean()
    print(str(frequency_names[frequency_count]) + ' average effect size = ' + str(average_effect_size_for_frequency))

print('  ')
print('Minimum effect size of amplitude vs. blackout (Cohen''s d) across electrodes: ')
print(' ')
for frequency_count in range(0,6):
    average_effect_size_for_frequency = effect_size_amplitude_compared_to_blackout[electrodes_to_use,frequency_count].mean()
    min_effect_size_for_frequency = min(effect_size_amplitude_compared_to_blackout[electrodes_to_use,frequency_count])
    print(str(frequency_names[frequency_count]) + ' min effect size = ' + str(min_effect_size_for_frequency))

    
        
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

Z_scores_amplitude_difference_compared_to_Pz_SIGI = np.zeros([8,6])
p_values_amplitude_difference_compared_to_Pz_SIGI = np.zeros([8,6])
effect_size_amplitude_difference_compared_to_Pz_SIGI = np.zeros([8,6])

for electrode in range(0,8):  #'VEOG', 'blink', 'P3', 'P4', 'O2', 'Pz', 'O1', 'HEOG'
    
    print(' ')
    print('Electrode ' +  electrode_names[electrode])
    print(' ')
      
    for frequency_count in range(0,6):
        

        
        all_Pz_SIGI_amplitude_differences = SIGI_amplitudes[:,frequency_count,5,1] - SIGI_amplitudes[:,frequency_count,5,0] # walking minus standing
       
        all_amplitude_differences = SSVEP_amplitudes[:,frequency_count,electrode,1] - SSVEP_amplitudes[:,frequency_count,electrode,0] # walking minus standing
        
        Z_score = functions.group_permutation_test(all_amplitude_differences, all_Pz_SIGI_amplitude_differences)

        p_value_one_sided = scipy.stats.norm.sf(abs(Z_score)) #one-sided
        
        p_value_two_sided = scipy.stats.norm.sf(abs(Z_score))*2 #twosided

        cohens_d = functions.cohens_d(all_amplitude_differences, all_Pz_SIGI_amplitude_differences)

        print(' ')
        print(str(frequencies_to_use[frequency_count]) + ' Hz')
        print('p = ' + str(p_value_two_sided))
        print('cohens d = ' + str(cohens_d))
        print('  ')        
       
        Z_scores_amplitude_difference_compared_to_Pz_SIGI[electrode,frequency_count] = Z_score
        p_values_amplitude_difference_compared_to_Pz_SIGI[electrode,frequency_count] = p_value_one_sided
        effect_size_amplitude_difference_compared_to_Pz_SIGI[electrode,frequency_count] = cohens_d        


# read out average effect sizes
print('  ')
print('Average effect size of amplitude difference vs. Signal generator (Cohen''s d) across electrodes: ')
print(' ')
frequency_names = ('30 Hz', '35 Hz', '40 Hz', '45 Hz', '50 Hz', '55 Hz', 'Signal \n Generator \n (40 Hz)')    
electrodes_to_use = (2,3,5,6,4) #  ('VEOG', 'blink', 'P3', 'P4', 'O2', 'Pz', 'O1', 'HEOG', 'x_dir', 'y_dir', 'z_dir')
electrodes_used_names = ('P3', 'P4', 'Pz', 'O1', 'O2')

for frequency_count in range(0,6):
    
    average_effect_size_for_frequency = effect_size_amplitude_difference_compared_to_Pz_SIGI[electrodes_to_use,frequency_count].mean()
    print(str(frequency_names[frequency_count]) + ' average effect size = ' + str(average_effect_size_for_frequency))

       






## Z scores phase shift vs. self split phase (from walking condition)

sig_cutoff = 1.96

phase_Z_scores = np.zeros([8,6])

frequency_names = ('30 Hz', '35 Hz', '40 Hz', '45 Hz', '50 Hz', '55 Hz')

print('  ')
print('Significant walking-standing absolute PHASE differences:')

for frequency in range(0,6):
    print(' ')
    for electrode in range(0,8):

        walking_self_split_phase_scores = all_mean_self_absolute_phase_shifts[:, frequency, electrode, 1] # condition 1 = walking
        
        walking_standing_phase_scores = SSVEP_phase_scores[:,frequency,electrode]     
        
        Z_score = functions.group_permutation_test(walking_standing_phase_scores, walking_self_split_phase_scores)         
    
        p_value_one_sided = scipy.stats.norm.sf(abs(Z_score)) #one-sided
    
        phase_Z_scores[electrode, frequency] = Z_score
    
        if Z_score > sig_cutoff:
            print(frequency_names[frequency] + '  ' + electrode_names[electrode] + '  Z score = ' + str(Z_score) + '  p = ' + str(p_value_one_sided))







## Z scores amplitude walking-standing difference vs. self split amplitude (from walking condition)

sig_cutoff = 1.96

phase_Z_scores = np.zeros([8,6])

frequency_names = ('30 Hz', '35 Hz', '40 Hz', '45 Hz', '50 Hz', '55 Hz')

print('  ')
print('Significant walking-standing absolute AMPLITUDE differences:')

for frequency in range(0,6):
    print(' ')
    for electrode in range(0,8):

        walking_self_split_amplitude_scores = self_split_amplitude_differences[:, frequency, electrode, 1] # condition 1 = walking
        
        walking_amplitudes =  SSVEP_amplitudes[:,frequency,electrode,1]
        standing_amplitudes =  SSVEP_amplitudes[:,frequency,electrode,0]
        
        walking_standing_amplitude_differences = walking_amplitudes - standing_amplitudes 
        
        Z_score = functions.group_permutation_test(walking_standing_amplitude_differences, walking_self_split_amplitude_scores)         
    
        p_value_one_sided = scipy.stats.norm.sf(abs(Z_score)) #one-sided
    
        phase_Z_scores[electrode, frequency] = Z_score
    
        if Z_score > sig_cutoff:
            print(frequency_names[frequency] + '  ' + electrode_names[electrode] + '  Z score = ' + str(Z_score) + '  p = ' + str(p_value_one_sided))











###########  Analysis of Waveform shape across frequencies #########





# ## get the correlation between the SSVEP at one frequency and the time warped SSVEP at the next frequency

# max_correlations_with_other_frequencies = np.zeros([10,8,2,6,6])


# for subject in range(0,10):
#     for electrode in range(0,8):
#         for condition in range(0,2):

#             for frequency_1 in range(0,6): 
#                 for frequency_2 in range(0,6): 
        
        
#                     period_frequency_1 = int(np.round(sample_rate/frequencies_to_use[frequency_1]))             
    
#                     SSVEP_1 = all_SSVEPs[subject,frequency_1,electrode,condition,0:period_frequency_1] 
        
#                     period_frequency_2 = int(np.round(sample_rate/frequencies_to_use[frequency_2]))             
    
#                     SSVEP_2 = all_SSVEPs[subject,frequency_2,electrode,condition,0:period_frequency_2] 
    
#                     time_warped_SSVEP = functions.time_warp_SSVEPs(SSVEP_1, SSVEP_2)
                    
#                     # get the cross correlation with the lowest frequency SSVEP
#                     if period_frequency_1 >= period_frequency_2:
#                         max_correlation = functions.max_correlation(SSVEP_1, time_warped_SSVEP)
#                     elif period_frequency_1 < period_frequency_2:
#                         max_correlation = functions.max_correlation(SSVEP_2, time_warped_SSVEP)
                        
#                     max_correlations_with_other_frequencies[subject,electrode,condition, frequency_1,frequency_2] = max_correlation
                    
     
                    
     
        
     
# ## average the correlation and plot in a grid
# correlation_cutoff = 0.4
 
# standing_or_walking = 1 # 0 = standing, 1 = walking
   
# electrodes_to_use = (2,3,4,5,6)
# grand_average_correlation_grid_matrix = np.zeros([len(electrodes_to_use),6,6])

# electrode_count = 0  

# for electrode in electrodes_to_use: ##'VEOG', 'blink', 'P3', 'P4', 'O2', 'Pz', 'O1', 'HEOG', 'x_dir', 'y_dir', 'z_dir'   

#     print(electrode_names[electrode])

#     # check which subjects have a significantly high signal to noise ratio
#     subjects_to_use = []
#     for subject in range(0,10):
        
#         standing_self_correlations_all_frequencies = all_mean_self_split_correlations[subject,:,electrode,0]  # subject, frequency, electrode, condition
#         walking_self_correlations_all_frequencies = all_mean_self_split_correlations[subject,:,electrode,1]  # subject, frequency, electrode, condition
       
#         if all(standing_self_correlations_all_frequencies > correlation_cutoff) and all(walking_self_correlations_all_frequencies > correlation_cutoff):
#             subjects_to_use.append(subject)
            

#     # plt.figure()
#     # plt.title(electrode_names[electrode])

#     correlations_all_subjects_standing = max_correlations_with_other_frequencies[:,electrode,0, :,:]
#     correlations_all_subjects_walking = max_correlations_with_other_frequencies[:,electrode,1, :,:]
    
#     # average only the frequencies and subjects which have a self-correlation above the cutoff value
#     average_correlations_grid_standing = np.zeros([6,6])
#     average_correlations_grid_walking = np.zeros([6,6])
#     for frequency_1 in range(0,6):
#         for frequency_2 in range(0,6):
            
#             # standing
#             subjects_to_use = []
#             for subject in range(0,10):
#                 if all_mean_self_split_correlations[subject,frequency_1,electrode,0] > correlation_cutoff and all_mean_self_split_correlations[subject,frequency_2,electrode,0] > correlation_cutoff:
#                     subjects_to_use.append(subject)
#             average_correlations_grid_standing[frequency_1, frequency_2] = correlations_all_subjects_standing[subjects_to_use,frequency_1, frequency_2].mean(axis=0)
#            # print(subjects_to_use)   
            
#             # walking
#             subjects_to_use = []
#             for subject in range(0,10):
#                 if all_mean_self_split_correlations[subject,frequency_1,electrode,1] > correlation_cutoff and all_mean_self_split_correlations[subject,frequency_2,electrode,1] > correlation_cutoff:
#                     subjects_to_use.append(subject)
#             average_correlations_grid_walking[frequency_1, frequency_2] = correlations_all_subjects_standing[subjects_to_use,frequency_1, frequency_2].mean(axis=0)
#             #print(subjects_to_use)     
    
#     # average all subjects
#     # average_correlations_grid_standing = correlations_all_subjects_standing.mean(axis=0)
#     # average_correlations_grid_walking = correlations_all_subjects_walking.mean(axis=0)
    
    
#     # average walking and standing
#     #walking_standing_average_correlations_grid = (average_correlations_grid_standing + average_correlations_grid_walking)/2
#     # grand_average_correlation_grid_matrix[electrode_count,:,:] = walking_standing_average_correlations_grid

#     if standing_or_walking == 0:
#         grand_average_correlation_grid_matrix[electrode_count,:,:] = average_correlations_grid_standing
#     elif standing_or_walking == 1:
#         grand_average_correlation_grid_matrix[electrode_count,:,:] = average_correlations_grid_walking

#     electrode_count += 1
    
#     #plot
#     plt.figure()
    
    
#     if standing_or_walking == 0:
#         plt.title(electrode_names[electrode] + ' Standing')
#         plt.imshow(average_correlations_grid_standing)
#     elif standing_or_walking == 1:
#         plt.title(electrode_names[electrode] + ' Walking')
#         plt.imshow(average_correlations_grid_walking)
        
#     plt.colorbar()

#     plt.xticks(ticks = (0,1,2,3,4,5), labels = ('30 Hz', '35 Hz','40 Hz','45 Hz','50 Hz', '55 Hz'))
#     plt.yticks(ticks = (0,1,2,3,4,5), labels = ('30 Hz', '35 Hz','40 Hz','45 Hz','50 Hz', '55 Hz'))

#     plt.show()
    


# grand_average_correlations_grid = grand_average_correlation_grid_matrix.mean(axis=0)


# plt.figure()
# if standing_or_walking == 0:
#     plt.title('Average all electrodes Standing')
# elif standing_or_walking == 1:
#     plt.title('Average all electrodes Walking')
 

# plt.imshow(grand_average_correlations_grid)
# plt.colorbar()

# plt.xticks(ticks = (0,1,2,3,4,5), labels = ('30 Hz', '35 Hz','40 Hz','45 Hz','50 Hz', '55 Hz'))
# plt.yticks(ticks = (0,1,2,3,4,5), labels = ('30 Hz', '35 Hz','40 Hz','45 Hz','50 Hz', '55 Hz'))

# plt.show()
