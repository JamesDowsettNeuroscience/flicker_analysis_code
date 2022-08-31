#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 13:46:05 2022

@author: James Dowsett
"""


import numpy as np
import matplotlib.pyplot as plt

from flicker_analysis_package import functions

period = 25

time_vector = np.arange(0,period)

flicker_frequency = 40
sample_rate = 1000

phase_shift = 0

length = 1 # length in seconds for the FFT



for SSVEP_amplitude in (1,2): # create two SSVEPs one small (amplitude = 1) one big (amplitude= 2)

    one_cycle_SSVEP = SSVEP_amplitude * np.sin(2 * np.pi * flicker_frequency/sample_rate * (time_vector-phase_shift)) 
    
    
    # repeat the single cycle at the correct frequency
    
    number_of_flickers = int((100*sample_rate)/period) # number of times the simulated flicker will repeat in 100 seconds
    
    # empty array to put the simulated SSVEP into
    simulated_SSVEP_data = np.zeros([100 * sample_rate])
    
    # use tile to repeat the basic SSVEP
    simulated_SSVEP_data[0:number_of_flickers*period] = np.tile(one_cycle_SSVEP,number_of_flickers )
    
    if SSVEP_amplitude == 1:
        data_amplitude_1 = np.copy(simulated_SSVEP_data)
    elif SSVEP_amplitude == 2:
        data_amplitude_2 = np.copy(simulated_SSVEP_data)
    


simulated_triggers = np.arange(0, len(simulated_SSVEP_data)-period, period) # make triggers, stop one period length before the end


num_noise_amplitudes = 1000

SSVEP_amplitudes_1 = np.zeros([num_noise_amplitudes,])
SSVEP_amplitudes_2 = np.zeros([num_noise_amplitudes,])

amplitude_differences = np.zeros([num_noise_amplitudes,])

self_correlations_1 = np.zeros([num_noise_amplitudes,])
self_correlations_2 = np.zeros([num_noise_amplitudes,])

SNRs_1 = np.zeros([num_noise_amplitudes,])
SNRs_2 = np.zeros([num_noise_amplitudes,])

FFT_peaks_1 = np.zeros([num_noise_amplitudes,])
FFT_peaks_2 = np.zeros([num_noise_amplitudes,])

FFT_SNR_1 = np.zeros([num_noise_amplitudes,])
FFT_SNR_2 = np.zeros([num_noise_amplitudes,])


for noise_amplitude in range(0,num_noise_amplitudes):
    
    print(noise_amplitude)

    ## add noise to the data

    noise = np.random.rand(len(simulated_SSVEP_data),) * noise_amplitude
    
    simulated_data_1 = data_amplitude_1 + noise

    
    noise = np.random.rand(len(simulated_SSVEP_data),) * noise_amplitude
    
    simulated_data_2 = data_amplitude_2 + noise
    
    
    
   
    # make the SSVEPs
    SSVEP_1 = functions.make_SSVEPs(simulated_data_1, simulated_triggers, period)
    SSVEP_2 = functions.make_SSVEPs(simulated_data_2, simulated_triggers, period)
    
    # get the peak to peak amplitudes
    SSVEP_amplitudes_1[noise_amplitude] = np.ptp(SSVEP_1)
    SSVEP_amplitudes_2[noise_amplitude] = np.ptp(SSVEP_2)
    
    # get the difference in amplitude between SSVEP_1 and SSVEP_2
    amplitude_differences[noise_amplitude] = np.ptp(SSVEP_2) - np.ptp(SSVEP_1)
    
    # get the correlation of a randon 50% split 
    self_correlations_1[noise_amplitude] = functions.compare_SSVEPs_split(simulated_data_1, simulated_triggers, period)
    self_correlations_2[noise_amplitude] = functions.compare_SSVEPs_split(simulated_data_2, simulated_triggers, period)
    
    # 
    copy_of_simulated_data_1 = np.copy(simulated_data_1)
    copy_of_simulated_data_2 = np.copy(simulated_data_2)
    
    SNRs_1[noise_amplitude] = functions.SNR_random(copy_of_simulated_data_1, simulated_triggers, period)
    SNRs_2[noise_amplitude] = functions.SNR_random(copy_of_simulated_data_2, simulated_triggers, period)
    
    ## evoked FFT
    evoked_FFT_spectrum_1 = functions.evoked_fft(simulated_data_1, simulated_triggers, length, sample_rate)
    evoked_FFT_spectrum_2 = functions.evoked_fft(simulated_data_2, simulated_triggers, length, sample_rate)
    
    
    FFT_peaks_1[noise_amplitude] = evoked_FFT_spectrum_1[flicker_frequency]
    FFT_peaks_2[noise_amplitude] = evoked_FFT_spectrum_2[flicker_frequency]
    
    frequency_noise_1 = (np.concatenate([evoked_FFT_spectrum_1[flicker_frequency-10:flicker_frequency-2], evoked_FFT_spectrum_1[flicker_frequency+2:flicker_frequency+10]])).mean()
    
    FFT_SNR_1[noise_amplitude] = evoked_FFT_spectrum_1[flicker_frequency] / frequency_noise_1
    
    frequency_noise_2 = (np.concatenate([evoked_FFT_spectrum_2[flicker_frequency-10:flicker_frequency-2], evoked_FFT_spectrum_2[flicker_frequency+2:flicker_frequency+10]])).mean()
    
    FFT_SNR_2[noise_amplitude] = evoked_FFT_spectrum_2[flicker_frequency] / frequency_noise_2
    
   # plt.plot(SSVEP)
    # plt.subplot(1,2,1)
    # plt.plot(simulated_data_1[0:100])
    # plt.plot(data_amplitude_1[0:100])
    
    # plt.subplot(1,2,2)
    # plt.plot(evoked_FFT_spectrum_1)
    
    
    
    
## plot final results   
   
plt.figure() 
   
plt.subplot(2,2,1)
plt.title('Amplitudes and self correlation')

plt.plot(SSVEP_amplitudes_1)
plt.plot(SSVEP_amplitudes_2)

plt.plot(self_correlations_1)
plt.plot(self_correlations_2)

# plot zero line
plt.plot(np.arange(0, num_noise_amplitudes),np.zeros([num_noise_amplitudes,]),  '--k')   


#plt.plot(amplitude_differences)

plt.subplot(2,2,2)
plt.title('SNRs')

plt.plot(SNRs_1)
plt.plot(SNRs_2)


plt.subplot(2,2,3)
plt.title('FFT peaks')

plt.plot(FFT_peaks_1)
plt.plot(FFT_peaks_2)

plt.subplot(2,2,4)
plt.title('FFT SNRs')

plt.plot(FFT_SNR_1)
plt.plot(FFT_SNR_2)


######
# SNR with peak-to-peak of random shuffle seems to give random spikes in the SNR, maybe better to use peak-to-peak without a 5 data point average

# FFT peaks are on average the same amplitude as more noise is added, wheras peak-to-peak amplitudes go up with additional noise 

###  

smooth_window = 10

smoothed_1 = np.zeros([len(SSVEP_amplitudes_1)-smooth_window,])
smoothed_2 = np.zeros([len(SSVEP_amplitudes_1)-smooth_window,])

smoothed_fft_1 = np.zeros([len(SSVEP_amplitudes_1)-smooth_window,])
smoothed_fft_2 = np.zeros([len(SSVEP_amplitudes_1)-smooth_window,])

for k in range(smooth_window,len(SSVEP_amplitudes_1)-smooth_window):
    
    smoothed_1[k] = SSVEP_amplitudes_1[k-smooth_window:k+smooth_window].mean()
    smoothed_2[k] = SSVEP_amplitudes_2[k-smooth_window:k+smooth_window].mean()
    
    smoothed_fft_1[k] = FFT_SNR_1[k-smooth_window:k+smooth_window].mean()
    smoothed_fft_2[k] = FFT_SNR_2[k-smooth_window:k+smooth_window].mean()
    
    
    
plt.figure()
plt.plot(smoothed_1)
plt.plot(smoothed_2)

plt.figure()
plt.title('smoothed FFT SNR')
plt.plot(smoothed_fft_1)
plt.plot(smoothed_fft_2)

























