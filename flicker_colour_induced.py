# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 23:58:08 2021

@author: Jorge Estudillo (TUM)

If something on this code is not working or seems sketchy,
please feel free to contact me: estudillolopezjorge@gmail.com
"""

import numpy as np
from general_functions import load_data, filter_EOG_and_EEG, FFT_plots, FFT_Master_pipeline, SNR # import defined functions

subject, file_name, data, VEOG_data, HEOG_data, electrode_name = load_data(1,0,'Oz')

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
fft_matrix = np.zeros([flicker_time,length_of_segment*5]) # zero matrix for all original data
harmonics = 3 # number of harmonics of interest (1 fundamental + 2 harmonics)
subharmonics = 4 # number of subharmonics (calculated as 1/(2+subharmonics), max now is 7)
lp_data = np.copy(data) # exact copy of original data. without this, alterations due to interpolation will also affect original data
sort_count = 0

for freq_count in range(len(flicker_periods)): # loop through each frequency
    
    flicker_period = flicker_periods[freq_count]
    for colour in ('red', 'green', 'blue', 'white'): # loops through each colour
        print(colour)
        print('Loading data...')
        all_triggers = np.load(file_name + '_all_' + colour + '_triggers.npy') # load triggers specific for each colour
        print('Done')
        diff_trigger_time = np.diff(all_triggers) # calculates the difference between trigger indices. this gives us a vector
        # of flicker_period * 5 values most of the time, which helps us determine which flicker freq. we are analyzing
        first_trigger = np.array(np.where(diff_trigger_time == flicker_period*5)) # every time a flicker freq. has been detected
        first_trigger = first_trigger[0] # we only keep the vector of trigger indices (np.where outputs more data we dont use)
        start_time = all_triggers[first_trigger[0]] # this first trigger for the selected freq. is passed as an index of all triggers
        
        ## LINEAR INTERPOLATION of all data, for each colour and flicker period
        segment_matrix = np.zeros([len(first_trigger), flicker_period*5]) # zeros matrix to save interpolated segments
        # this is only used to check if interpolation is correct. For this, uncomment the corresponding section
        for k in range(len(first_trigger)): # iterating through all triggers for selected flicker freq.
            # iterates through the trigger indices for selected flicker freq.
            trigger_time = all_triggers[first_trigger[k]]# use this as the index of all the triggers
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
            
        print('Extracting data...')
        # extracts a whole minute of data, for each flicker period and colour, for original and interpolated data
        flicker_data = data[start_time:start_time+flicker_time*length_of_segment*sample_rate]
        flicker_lp = lp_data[start_time:start_time+flicker_time*length_of_segment*sample_rate]
        print('Done')

        ## COLOUR + FLICKER SORTING
        # basically checks in which colour and flicker period we were currently analyzing, and copies results in a corresponding vector
        # this occurs for both original and interpolated data
        if colour == 'red':
            if flicker_period*5 == 165:
                red_33ms_data = np.copy(flicker_data)
                red_33ms_lp = np.copy(flicker_lp)
            if flicker_period*5 == 140:
                red_28ms_data = np.copy(flicker_data)
                red_28ms_lp = np.copy(flicker_lp)
            if flicker_period*5 == 125:
                red_25ms_data = np.copy(flicker_data)
                red_25ms_lp = np.copy(flicker_lp)
            if flicker_period*5 == 110:
                red_22ms_data = np.copy(flicker_data)
                red_22ms_lp = np.copy(flicker_lp)
            if flicker_period*5 == 90:
                red_18ms_data = np.copy(flicker_data)
                red_18ms_lp = np.copy(flicker_lp)
            if flicker_period*5 == 85:
                red_17ms_data = np.copy(flicker_data)
                red_17ms_lp = np.copy(flicker_lp)
        if colour == 'green':
            if flicker_period*5 == 165:
                green_33ms_data = np.copy(flicker_data)
                green_33ms_lp = np.copy(flicker_lp)
            if flicker_period*5 == 140:
                green_28ms_data = np.copy(flicker_data)
                green_28ms_lp = np.copy(flicker_lp)
            if flicker_period*5 == 125:
                green_25ms_data = np.copy(flicker_data)
                green_25ms_lp = np.copy(flicker_lp)
            if flicker_period*5 == 110:
                green_22ms_data = np.copy(flicker_data)
                green_22ms_lp = np.copy(flicker_lp)
            if flicker_period*5 == 90:
                green_18ms_data = np.copy(flicker_data)
                green_18ms_lp = np.copy(flicker_lp)
            if flicker_period*5 == 85:
                green_17ms_data = np.copy(flicker_data)
                green_17ms_lp = np.copy(flicker_lp)
        if colour == 'blue':
            if flicker_period*5 == 165:
                blue_33ms_data = np.copy(flicker_data)
                blue_33ms_lp = np.copy(flicker_lp)
            if flicker_period*5 == 140:
                blue_28ms_data = np.copy(flicker_data)
                blue_28ms_lp = np.copy(flicker_lp)
            if flicker_period*5 == 125:
                blue_25ms_data = np.copy(flicker_data)
                blue_25ms_lp = np.copy(flicker_lp)
            if flicker_period*5 == 110:
                blue_22ms_data = np.copy(flicker_data)
                blue_22ms_lp = np.copy(flicker_lp)
            if flicker_period*5 == 90:
                blue_18ms_data = np.copy(flicker_data)
                blue_18ms_lp = np.copy(flicker_lp)
            if flicker_period*5 == 85:
                blue_17ms_data = np.copy(flicker_data)
                blue_17ms_lp = np.copy(flicker_lp)
        if colour == 'white':
            if flicker_period*5 == 165:
                white_33ms_data = np.copy(flicker_data)
                white_33ms_lp = np.copy(flicker_lp)
            if flicker_period*5 == 140:
                white_28ms_data = np.copy(flicker_data)
                white_28ms_lp = np.copy(flicker_lp)
            if flicker_period*5 == 125:
                white_25ms_data = np.copy(flicker_data)
                white_25ms_lp = np.copy(flicker_lp)
            if flicker_period*5 == 110:
                white_22ms_data = np.copy(flicker_data)
                white_22ms_lp = np.copy(flicker_lp)
            if flicker_period*5 == 90:
                white_18ms_data = np.copy(flicker_data)
                white_18ms_lp = np.copy(flicker_lp)
            if flicker_period*5 == 85:
                white_17ms_data = np.copy(flicker_data)
                white_17ms_lp = np.copy(flicker_lp)
                
        # COLOUR_FLICKER_SORTING STILL NOT WORKING        
        # if colour == 'red':
        #     sort_count, _ = colour_flicker_sorting(flicker_period, flicker_data, sort_count)
        #     if sort_count == 6:
        #         sort_count = 0
        #         red_33ms_data, red_28ms_data, red_25ms_data, red_22ms_data, red_18ms_data, red_17ms_data = colour_flicker_sorting(flicker_period, flicker_data, sort_count)
        # if colour == 'green':
        #     sort_count, _ = colour_flicker_sorting(flicker_period, flicker_data, sort_count)
        #     if sort_count == 6:
        #         sort_count = 0
        #         green_33ms_data, green_28ms_data, green_25ms_data, green_22ms_data, green_18ms_data, green_17ms_data = colour_flicker_sorting(flicker_period, flicker_data, sort_count)
        # if colour == 'blue':
        #     sort_count, _ = colour_flicker_sorting(flicker_period, flicker_data, sort_count)
        #     if sort_count == 6:
        #         sort_count = 0
        #         blue_33ms_data, blue_28ms_data, blue_25ms_data, blue_22ms_data, blue_18ms_data, blue_17ms_data = colour_flicker_sorting(flicker_period, flicker_data, sort_count)
        # if colour == 'white':
        #     sort_count, _ = colour_flicker_sorting(flicker_period, flicker_data, sort_count)
        #     if sort_count == 6:
        #         sort_count = 0
        #         white_33ms_data, white_28ms_data, white_25ms_data, white_22ms_data, white_18ms_data, white_17ms_data = colour_flicker_sorting(flicker_period, flicker_data, sort_count)
                
        # CHECK TRIGGER CORRECTNESS        
        # trigger_time_series = np.zeros([len(data)])
        # for trigger in all_triggers:
        #     trigger_time_series[trigger] = 1
        # plt.figure()
        # plt.plot(trigger_time_series)

# stacking all original data into a matrix, to use with future functions
blue_data = np.array([blue_33ms_data,blue_28ms_data,blue_25ms_data,blue_22ms_data,blue_18ms_data,blue_17ms_data])
blue_ffts = FFT_Master_pipeline(blue_data, flicker_time, length_of_segment, 'i') # calculating FFTs for all blue original data
# stacking all interpolated data into a matrix, to use with future functions
blue_lp = np.array([blue_33ms_lp,blue_28ms_lp,blue_25ms_lp,blue_22ms_lp,blue_18ms_lp,blue_17ms_lp])
blue_lp_ffts = FFT_Master_pipeline(blue_lp, flicker_time, length_of_segment, 'i') # calculating FFTs for all blue interpolated data

# stacking all original data into a matrix, to use with future functions
red_data = np.array([red_33ms_data,red_28ms_data,red_25ms_data,red_22ms_data,red_18ms_data,red_17ms_data])
red_ffts = FFT_Master_pipeline(red_data, flicker_time, length_of_segment, 'i') # calculating FFTs for all red original data
# stacking all interpolated data into a matrix, to use with future functions
red_lp = np.array([red_33ms_lp,red_28ms_lp,red_25ms_lp,red_22ms_lp,red_18ms_lp,red_17ms_lp])
red_lp_ffts = FFT_Master_pipeline(red_lp, flicker_time, length_of_segment, 'i') # calculating FFTs for all red interpolated data

# stacking all original data into a matrix, to use with future functions
green_data = np.array([green_33ms_data,green_28ms_data,green_25ms_data,green_22ms_data,green_18ms_data,green_17ms_data])
green_ffts = FFT_Master_pipeline(green_data, flicker_time, length_of_segment, 'i') # calculating FFTs for all green original data
# stacking all interpolated data into a matrix, to use with future functions
green_lp = np.array([green_33ms_lp,green_28ms_lp,green_25ms_lp,green_22ms_lp,green_18ms_lp,green_17ms_lp])
green_lp_ffts = FFT_Master_pipeline(green_lp, flicker_time, length_of_segment, 'i') # calculating FFTs for all green interpolated data

# stacking all original data into a matrix, to use with future functions
white_data = np.array([white_33ms_data,white_28ms_data,white_25ms_data,white_22ms_data,white_18ms_data,white_17ms_data])
white_ffts = FFT_Master_pipeline(white_data, flicker_time, length_of_segment, 'i') # calculating FFTs for all white original data
# stacking all interpolated data into a matrix, to use with future functions
white_lp = np.array([white_33ms_lp,white_28ms_lp,white_25ms_lp,white_22ms_lp,white_18ms_lp,white_17ms_lp])
white_lp_ffts = FFT_Master_pipeline(white_lp, flicker_time, length_of_segment, 'i') # calculating FFTs for all white interpolated data

## FIND PEAKS AND CALCULATE SNR
all_ffts = np.array([blue_ffts,red_ffts,green_ffts,white_ffts]) # stacks all original FFTs 
np.save(file_name + '_' + electrode_name + '_all_ffts',all_ffts) # saves all original FFTs

all_lp_ffts = np.array([blue_lp_ffts,red_lp_ffts,green_lp_ffts,white_lp_ffts]) # stacks all interpolated FFTs

# calculates peak amplitudes, peak indices, and SNR for all colours, with a window of 2 Hz
all_peaks, all_indices, SNR_ffts = SNR(all_ffts,flicker_periods,length_of_segment,2)

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

# saves all data and interpolated data, separated by colour
print('Saving data...')
colour_matrix = ['blue', 'red', 'green', 'white']
all_data = np.array([blue_data,red_data,green_data,white_data])
all_lp = np.array([blue_lp,red_lp,green_lp,white_lp])
for i in range(len(all_data)):
    np.save(file_name + '_' + electrode_name + '_' + colour_matrix[i] + '_data', all_data[i])
    np.save(file_name + '_' + electrode_name + '_' + colour_matrix[i] + '_interpolated_data', all_lp[i])
print('Done')