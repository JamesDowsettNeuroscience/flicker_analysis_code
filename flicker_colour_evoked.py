# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 00:42:52 2021

@author: Jorge Estudillo (TUM)

If something on this code is not working or seems sketchy,
please feel free to contact me: estudillolopezjorge@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from general_functions import load_data, filter_EOG_and_EEG, FFT_plots, FFT_Master_pipeline, SNR # import defined functions

subject, file_name, data, VEOG_data, HEOG_data, electrode_name = load_data(1,0,'Oz') # load EOG and EEG data for specified electrode

flicker_periods = [33, 28, 25, 22, 18, 17] # flicker period defined in secs. time separation between flickers
offset = 1 # shift the SSVEP by some small amount so the trigger is not at the edge of the segment

# Load data
print('  ')
print('Loading EOG data ...')
print('  ')

# filter EEG data. EOG data can be also filtered but its not used for FFT computations, so I set their flags to 0
VEOG_data, HEOG_data, data, sample_rate = filter_EOG_and_EEG(VEOG_data, HEOG_data, data, 5000, 0.1, 5, 10, 100, EOG_HPF_flag=False, EOG_LPF_flag=False)

sample_rate = 5000 # predefined sampling rate of experiment
length_of_segment = 1 # segment duration (seconds)
flicker_time = int((300000/length_of_segment)/sample_rate) # total time of stimulus, calculated as the total length in a minute
# and divided by the sample rate and the length in secs. for length = 1, flicker_time = 60 secs. Bigger lengths make smaller times
data_matrix = np.zeros([flicker_time, length_of_segment*sample_rate]) # zero matrix for extracted evoked original data
data_matrix2 = np.zeros([flicker_time, length_of_segment*sample_rate]) # zero matrix for extracted evoked interpolated (lp) data

harmonics = 3 # number of harmonics of interest (1 fundamental + 2 harmonics)
subharmonics = 4 # number of subharmonics (calculated as 1/(2+subharmonics), max now is 7)
lp_data = np.copy(data) # exact copy of original data. without this, alterations due to interpolation will also affect original data

for freq_count in range(len(flicker_periods)): # loop through each frequency
    
    flicker_period = flicker_periods[freq_count]
    for colour in ('red', 'green', 'blue', 'white'): # loops through each colour
        print(colour)
        print('Loading data...')
        all_triggers = np.load(file_name + '_all_' + colour + '_triggers.npy') # load triggers specific for each colour
        # these colour triggers are an output of the "sort_triggers_colour" code. Please run this code before attempting to run this
        print('Done')
        diff_trigger_time = np.diff(all_triggers) # calculates the difference between trigger indices. this gives us a vector
        # of flicker_period * 5 values most of the time, which helps us determine which flicker freq. we are analyzing
        first_trigger = np.array(np.where(diff_trigger_time == flicker_period*5)) # every time a flicker freq. has been detected
        first_trigger = first_trigger[0] # we only keep the vector of trigger indices (np.where outputs more data we dont use)
        start_time = all_triggers[first_trigger[0]] # this first trigger for the selected freq. is passed as an index of all triggers
        print('Extracting data...')
                
        ## LINEAR INTERPOLATION of all data, for each colour and flicker period
        segment_matrix = np.zeros([len(first_trigger), flicker_period*5]) # zeros matrix to save interpolated segments
        # this is only used to check if interpolation is correct. For this, uncomment the corresponding section
        for k in range(len(first_trigger)): # iterating through all triggers for selected flicker freq.
            # iterates through the trigger indices for selected flicker freq.
            trigger_time = all_triggers[first_trigger[k]] # use this as the index of all the triggers
            # a segment of the whole timepoints obtained by one single trigger
            segment = lp_data[trigger_time-offset:trigger_time+(flicker_period*5)-offset] # extracted from the copied original data
            # because of how the shutter glasses work, there are to artifacts in the original data that should be interpolated
            # these artifacts occur exactly at the beginning of the segment (flicker ON) and exactly at halfway segment (flicker OFF)
            second_artifact = int(np.ceil((flicker_period*5)/2)) # calculates where the second artifact is, based on duration of flicker
            # artifact duration has been empirically determined to be = 12
            segment[0:12] = np.linspace(segment[0], segment[12], num=12) # interpolates data of the 1st artifact
            # in a similar fashion, interpolates data of the 2nd artifact
            segment[second_artifact:second_artifact+12] = np.linspace(segment[second_artifact], segment[second_artifact+12], num=12)
            segment_matrix[k,:] = segment # check for correct interpolation in next routine
            lp_data[trigger_time-offset:trigger_time+(flicker_period*5)-offset] = segment # pastes interpolated segment
            # exactly where the original was, in the lp_data, keeping the original untouched
        
        # ## CHECK FOR CORRECT INTERPOLATION OF DATA
        # if flicker_period*5 == 90: # please adjust to desired flicker period
        #     if colour == 'red': # please adjust to desired colour
        #         plot_color = 'r'
        #         mean_segment_matrix = segment_matrix.mean(axis=0)
        #         mean_segment_matrix[0:12] = np.linspace(mean_segment_matrix[0], mean_segment_matrix[12], num=12)
        #         mean_segment_matrix[second_artifact:second_artifact+12] = np.linspace(mean_segment_matrix[second_artifact], mean_segment_matrix[second_artifact+12], num=12)
        #         plt.figure(1000)
        #         plt.plot(mean_segment_matrix, plot_color)
        
        for i in range(flicker_time): # extracts phase-locked segments. number of segments depends on length (thus total time)
            # obtains a segment of the desired length, phase-locked with the start time, that iterates across all colour triggers
            evoked_data = data[start_time:start_time+length_of_segment*sample_rate]
            data_matrix[i,:] = evoked_data # saves this original segment 
            
            # obtains a segment of the desired length, phase-locked with the start time, that iterates across all colour triggers
            evoked_data2 = lp_data[start_time:start_time+length_of_segment*sample_rate] # but now for the interpolated data
            data_matrix2[i,:] = evoked_data2 # saves this interpolated segment
            
            # creates a vector of possible next triggers that would be phase-locked, ignores the segment we already extracted
            next_trigger = np.array(np.where(all_triggers > start_time+length_of_segment*sample_rate))
            next_trigger = next_trigger[0] # we only take the first next trigger, as its the closest to the last extracted segment
            if len(next_trigger) == 0: # checks if vector is empty. this happens when there are no 5000 points after last trigger
                continue # if it is empty, break the loop and continue to sorting
            start_time = all_triggers[next_trigger[0]] # uses this next trigger as the starting time for next iteration
            
        print('Done')
        
        ## COLOUR + FLICKER SORTING
        # basically checks in which colour and flicker period we were currently analyzing, and copies results in a corresponding vector
        # this occurs for both original and interpolated data
        if colour == 'red':
            if flicker_period*5 == 165:
                red_33ms_data = np.copy(data_matrix)
                red_33ms_lp = np.copy(data_matrix2)
            if flicker_period*5 == 140:
                red_28ms_data = np.copy(data_matrix)
                red_28ms_lp = np.copy(data_matrix2)
            if flicker_period*5 == 125:
                red_25ms_data = np.copy(data_matrix)
                red_25ms_lp = np.copy(data_matrix2)
            if flicker_period*5 == 110:
                red_22ms_data = np.copy(data_matrix)
                red_22ms_lp = np.copy(data_matrix2)
            if flicker_period*5 == 90:
                red_18ms_data = np.copy(data_matrix)
                red_18ms_lp = np.copy(data_matrix2)
            if flicker_period*5 == 85:
                red_17ms_data = np.copy(data_matrix)
                red_17ms_lp = np.copy(data_matrix2)
        if colour == 'green':
            if flicker_period*5 == 165:
                green_33ms_data = np.copy(data_matrix)
                green_33ms_lp = np.copy(data_matrix2)
            if flicker_period*5 == 140:
                green_28ms_data = np.copy(data_matrix)
                green_28ms_lp = np.copy(data_matrix2)
            if flicker_period*5 == 125:
                green_25ms_data = np.copy(data_matrix)
                green_25ms_lp = np.copy(data_matrix2)
            if flicker_period*5 == 110:
                green_22ms_data = np.copy(data_matrix)
                green_22ms_lp = np.copy(data_matrix2)
            if flicker_period*5 == 90:
                green_18ms_data = np.copy(data_matrix)
                green_18ms_lp = np.copy(data_matrix2)
            if flicker_period*5 == 85:
                green_17ms_data = np.copy(data_matrix)
                green_17ms_lp = np.copy(data_matrix2)
        if colour == 'blue':
            if flicker_period*5 == 165:
                blue_33ms_data = np.copy(data_matrix)
                blue_33ms_lp = np.copy(data_matrix2)
            if flicker_period*5 == 140:
                blue_28ms_data = np.copy(data_matrix)
                blue_28ms_lp = np.copy(data_matrix2)
            if flicker_period*5 == 125:
                blue_25ms_data = np.copy(data_matrix)
                blue_25ms_lp = np.copy(data_matrix2)
            if flicker_period*5 == 110:
                blue_22ms_data = np.copy(data_matrix)
                blue_22ms_lp = np.copy(data_matrix2)
            if flicker_period*5 == 90:
                blue_18ms_data = np.copy(data_matrix)
                blue_18ms_lp = np.copy(data_matrix2)
            if flicker_period*5 == 85:
                blue_17ms_data = np.copy(data_matrix)
                blue_17ms_lp = np.copy(data_matrix2)
        if colour == 'white':
            if flicker_period*5 == 165:
                white_33ms_data = np.copy(data_matrix)
                white_33ms_lp = np.copy(data_matrix2)
            if flicker_period*5 == 140:
                white_28ms_data = np.copy(data_matrix)
                white_28ms_lp = np.copy(data_matrix2)
            if flicker_period*5 == 125:
                white_25ms_data = np.copy(data_matrix)
                white_25ms_lp = np.copy(data_matrix2)
            if flicker_period*5 == 110:
                white_22ms_data = np.copy(data_matrix)
                white_22ms_lp = np.copy(data_matrix2)
            if flicker_period*5 == 90:
                white_18ms_data = np.copy(data_matrix)
                white_18ms_lp = np.copy(data_matrix2)
            if flicker_period*5 == 85:
                white_17ms_data = np.copy(data_matrix)
                white_17ms_lp = np.copy(data_matrix2)
                
            # trigger_time_series = np.zeros([len(data)])
            # for trigger in all_triggers:
            #     trigger_time_series[trigger] = 1
            # plt.figure()
            # plt.plot(trigger_time_series)
        
np.save(file_name + '_' + electrode_name + '_all_interpolated_data',lp_data) # saves huge vector of interpolated data

# stacking all original data into a matrix, to use with future functions
blue_data = np.array([blue_33ms_data,blue_28ms_data,blue_25ms_data,blue_22ms_data,blue_18ms_data,blue_17ms_data])
blue_ffts = FFT_Master_pipeline(blue_data, flicker_time, length_of_segment, 'e') # calculating FFTs for all blue original data
# stacking all interpolated data into a matrix, to use with future functions
blue_lp = np.array([blue_33ms_lp,blue_28ms_lp,blue_25ms_lp,blue_22ms_lp,blue_18ms_lp,blue_17ms_lp])
blue_lp_ffts = FFT_Master_pipeline(blue_lp, flicker_time, length_of_segment, 'e') # calculating FFTs for all blue interpolated data

# stacking all original data into a matrix, to use with future functions
red_data = np.array([red_33ms_data,red_28ms_data,red_25ms_data,red_22ms_data,red_18ms_data,red_17ms_data])
red_ffts = FFT_Master_pipeline(red_data, flicker_time, length_of_segment, 'e') # calculating FFTs for all red original data
# stacking all interpolated data into a matrix, to use with future functions
red_lp = np.array([red_33ms_lp,red_28ms_lp,red_25ms_lp,red_22ms_lp,red_18ms_lp,red_17ms_lp])
red_lp_ffts = FFT_Master_pipeline(red_lp, flicker_time, length_of_segment, 'e') # calculating FFTs for all red interpolated data

# stacking all original data into a matrix, to use with future functions
green_data = np.array([green_33ms_data,green_28ms_data,green_25ms_data,green_22ms_data,green_18ms_data,green_17ms_data])
green_ffts = FFT_Master_pipeline(green_data, flicker_time, length_of_segment, 'e') # calculating FFTs for all green original data
# stacking all interpolated data into a matrix, to use with future functions
green_lp = np.array([green_33ms_lp,green_28ms_lp,green_25ms_lp,green_22ms_lp,green_18ms_lp,green_17ms_lp])
green_lp_ffts = FFT_Master_pipeline(green_lp, flicker_time, length_of_segment, 'e') # calculating FFTs for all green interpolated data

# stacking all original data into a matrix, to use with future functions
white_data = np.array([white_33ms_data,white_28ms_data,white_25ms_data,white_22ms_data,white_18ms_data,white_17ms_data])
white_ffts = FFT_Master_pipeline(white_data, flicker_time, length_of_segment, 'e') # calculating FFTs for all white original data
# stacking all interpolated data into a matrix, to use with future functions
white_lp = np.array([white_33ms_lp,white_28ms_lp,white_25ms_lp,white_22ms_lp,white_18ms_lp,white_17ms_lp])
white_lp_ffts = FFT_Master_pipeline(white_lp, flicker_time, length_of_segment, 'e') # calculating FFTs for all white interpolated data

## FIND PEAKS AND CALCULATE SNR
all_ffts = np.array([blue_ffts,red_ffts,green_ffts,white_ffts]) # stacks all original FFTs 
np.save(file_name + '_' + electrode_name + '_all_evoked_ffts_' + str(length_of_segment) + 's',all_ffts) # saves all original FFTs

all_lp_ffts = np.array([blue_lp_ffts,red_lp_ffts,green_lp_ffts,white_lp_ffts]) # stacks all interpolated FFTs
np.save(file_name + '_' + electrode_name + '_all_evoked_lp_ffts_' + str(length_of_segment) + 's',all_lp_ffts) # saves all interpolated FFTs

# calculates peak amplitudes, peak indices, and SNR for all colours, with a window of 2 Hz
all_peaks, all_indices, SNR_ffts = SNR(all_ffts,flicker_periods,length_of_segment,2)
np.save(file_name + '_' + electrode_name + '_all_evoked_SNR_' + str(length_of_segment) + 's',SNR_ffts) # saves all original SNRs

# plots all original and interpolated FFTs (compared), with original peaks
FFT_plots(blue_ffts, blue_lp_ffts, flicker_periods, harmonics, subharmonics, length_of_segment*sample_rate, colour='b', peak_indices=all_indices)
FFT_plots(red_ffts, red_lp_ffts, flicker_periods, harmonics, subharmonics, length_of_segment*sample_rate, colour='r', peak_indices=all_indices)
FFT_plots(green_ffts, green_lp_ffts, flicker_periods, harmonics, subharmonics, length_of_segment*sample_rate, colour='g', peak_indices=all_indices)
FFT_plots(white_ffts, white_lp_ffts, flicker_periods, harmonics, subharmonics, length_of_segment*sample_rate, colour='k', peak_indices=all_indices)

# plots all original FFTs -- no comparison, no peaks
# FFT_plots(blue_ffts, blue_lp_ffts, flicker_periods, harmonics, subharmonics, length_of_segment*sample_rate, colour='b', comparison=False)
# FFT_plots(red_ffts, red_lp_ffts, flicker_periods, harmonics, subharmonics, length_of_segment*sample_rate, colour='r', comparison=False)
# FFT_plots(green_ffts, green_lp_ffts, flicker_periods, harmonics, subharmonics, length_of_segment*sample_rate, colour='g', comparison=False)
# FFT_plots(white_ffts, white_lp_ffts, flicker_periods, harmonics, subharmonics, length_of_segment*sample_rate, colour='k', comparison=False)

## STILL TO WORK ON: GENERALIZE SNR CALCULATION FOR DIFFERENT LENGTH OF SEGMENTS, PEAKS ARE BEING MISCALCULATED

############ ANOTATION OF WRONG PEAKS -- ORIGINAL DATA: 1sec segments
## blue@36Hz fundamental (wrong peak), 2nd harmonic (outside window)
## blue@40Hz 1st harmonic (outside window)
## red@36Hz fundamental (outside window)
## red@40Hz 1st harmonic (outside window)
## red@45Hz fundamental (wrong peak and outside window), 2nd harmonic (outside window)
## red@56Hz 1st harmonic (wrong peak)
## green@40Hz harmonics (wrong peaks and outside window)
## green@56Hz 1st harmonic (wrong peak)
## white@36Hz 2nd harmonic (outside window)
## white@40Hz 1st harmonic (outside window)
## white@56Hz 1st harmonic (wrong peak)
## ALL SUBHARMONICS

