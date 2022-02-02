# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 23:58:08 2021

@author: Jorge Estudillo (TUM)

If something on this code is not working or seems sketchy,
please feel free to contact me: estudillolopezjorge@gmail.com
"""
import numpy as np
import mne
from scipy import signal, fftpack
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def load_data(subject, load_mode, electrode_name=''):
    """
    This function helps importing files and/or numpy arrays used in several experiments. Depending on what is
    getting imported, this function has several return parameters that are already set to be used in following codes.
    The load_mode parameter defines what is being imported. For more information check the load_mode parameter
    To import this function into your code, please use "from general_functions import load_data"
    Parameters
    ----------
    subject : INT
        Requires subject number to locate files
    electrode_name : STR
        Requires analysis electrode to locate file
    load_mode : INT
        Input 0 to load .npy data from one electrode and EOG data, "make_SSVEPs_colour" code
        Input 1 to load .vhdr file for mne including channel names, "load_data_single_electrode" code
        Input 2 to load .vmrk file and calls it f, "sort_triggers_colour" code (no electrode name needed)

    Returns
    -------
    data : NPARRAY
        Returns specified electrode data from subject
    VEOG_data : NPARRAY
        Returns vertical EOG data for blink detection
    HEOG_data : NPARRAY
        Returns horizontal EOG data for blink detection
    file_name : STR
        Returns file name for future reference
    subject : INT
        Returns subject number for future reference
    electrode_name : STR
        Returns electrode name for future reference
    raw : EEG mne data
        Returns row EEG data as read by mne module
    channel_names : mne data
        Returns channel number corresponding to electrode name
    electrode : mne data
        Returns channel index corresponding to electrode name

    """
    file_name = 'S' + str(subject) + '_colour'
    
    if load_mode == 0: ## entering .npy load mode
        electrode_data_file_name = file_name + '_' + electrode_name +'_data.npy'
        VEOG_data_file_name = file_name + '_VEOG_data.npy'
        HEOG_data_file_name = file_name + '_HEOG_data.npy'
        
        data = np.load(electrode_data_file_name)
        VEOG_data = np.load(VEOG_data_file_name)
        HEOG_data = np.load(HEOG_data_file_name)
    
        return subject, file_name, data, VEOG_data, HEOG_data, electrode_name
    
    elif load_mode == 1:  ## entering .vhdr load mode for electrode data
        # read the EEG data with the MNE function
        raw = mne.io.read_raw_brainvision(file_name + '.vhdr')
        
        # get the channel number corresponding to the electrode name
        channel_names = raw.info.ch_names
        electrode = channel_names.index(electrode_name)
        
        return subject, file_name, raw, channel_names, electrode, electrode_name
        
    elif load_mode == 2: ## entering .vmrk load mode for reading triggers
        ### read triggers
        f = open(file_name + '.vmrk') # open the .vmrk file, call it "f"
        
        return subject, file_name, f
    
def extract_mne_data(raw, electrode, file_name, electrode_name):
    """
    This functions helps extracting EEG raw data, VEOG and HEOG data as given by mne, saving the data as numpy array 
    files. All the necessary parameters for this function is given by load_data mode 1
    To import this function into your code, please use "from general_functions import extract_mne_data"
    Parameters
    ----------
    raw : EEG mne data
        raw EEG data as given by mne
    electrode : mne data
        channel index corresponding to electrode name
    file_name : STR
        Returns file name for future reference
    electrode : mne data
        Returns channel index corresponding to electrode name

    Returns
    -------
    None.

    """
    # Load data
    print('  ')
    print('Loading EOG data ...')
    print('  ')
    
    # extract the EOG data
    VEOG_data = np.array(raw[30,:], dtype=object) 
    VEOG_data = VEOG_data[0,]
    VEOG_data = VEOG_data.flatten()
    
    HEOG_data = np.array(raw[31,:], dtype=object) 
    HEOG_data = HEOG_data[0,]
    HEOG_data = HEOG_data.flatten()
    # extract the data for one electrode
    print('Loading electrode data ...')
    print('  ')
    
    data = np.array(raw[electrode,:], dtype=object) 
    data = data[0,]
    data = data.flatten()
    
    channel_names = raw.info.ch_names 
    channel_names[60] = 'Fpz' # correct names spelt wrong
    
    print('Saving ...')
    
    electrode_data_file_name = file_name + '_' + electrode_name +'_data'
    VEOG_data_file_name = file_name + '_VEOG_data'
    HEOG_data_file_name = file_name + '_HEOG_data'
    
    # save as a numpy array
    np.save(electrode_data_file_name, data)
    np.save(VEOG_data_file_name, VEOG_data)
    np.save(HEOG_data_file_name, HEOG_data)
    
    print('Done')
    
def filter_EOG_and_EEG(VEOG_data, HEOG_data, data, sample_rate, EOG_HPF_cutoff, EOG_LPF_cutoff, EEGdata_HPF_cutoff, 
                       EEGdata_LPF_cutoff, EOG_HPF_order=2, EOG_LPF_order=3, EEGdata_HPF_order=2, EEGdata_LPF_order=1, 
                       EOG_HPF_flag=True, EOG_LPF_flag=True, EEGdata_HPF_flag=True, EEGdata_LPF_flag=False):
    """
    This functions allows filtering of EOG and EEG (specified electrode) data as obtained from mne and loaded by the
    corresponding function. It requires the data to be filtered (VEOG, HEOG and electrode data) as well as the cutoff
    frequency for all filters (including LPF for EEG data which is turned off by default as this code section was
    originally commented). The order of all filters can be modified but, by default, they are set to the orders from the
    original code. Finally, all filters can be turned on/off with their respective flags but, by default, the commented
    filters (EEG LPF) is turned off while the rest is turned on. It also requires the sampling rate of the system.
    If EOG data is unavailable or not used, you can pass 0 as an argument for EOG parameters and turn their flags to False
    Parameters
    ----------
    VEOG_data : NPARRAY
        Vertical EOG data to be filtered. Obtained from load_data mode 0
    HEOG_data : NPARRAY
        Horizontal EOG data to be filtered. Obtained from load_data mode 0
    data : NPARRAY
        EEG data from specified electrode. Obtained from load_data mode 0
    sample_rate : INT
        Sampling rate of the system
    EOG_HPF_cutoff : FLOAT
        Cutoff frequency for high pass filter to be used with VEOG and HEOG data
    EOG_LPF_cutoff : FLOAT
        Cutoff frequency for low pass filter to be used with VEOG and HEOG data
    EEGdata_HPF_cutoff : FLOAT
        Cutoff frequency for high pass filter to be used with electrode data
    EEGdata_LPF_cutoff : FLOAT
        Cutoff frequency for low pass filter to be used with electrode data. This filter is off by default but you can set
        it on using a 100 Hz frequency
    EOG_HPF_order : INT, optional
        Order of the high pass filter to be used with VEOG and HEOG data. The default is 2.
    EOG_LPF_order : INT, optional
        Order of the low pass filter to be used with VEOG and HEOG data. The default is 3.
    EEGdata_HPF_order : INT, optional
        Order of the high pass filter to be used with electrode data. The default is 2.
    EEGdata_LPF_order : INT, optional
        Order of the low pass filter to be used with electrode data. The default is 1. This filter is off by default
    EOG_HPF_flag : BOOL, optional
        Turns on/off the high pass filter to be used with VEOG and HEOG data. The default is True (filter on)
    EOG_LPF_flag : BOOL, optional
        Turns on/off the low pass filter to be used with VEOG and HEOG data. The default is True (filter on)
    EEGdata_HPF_flag : BOOL, optional
        Turns on/off the high pass filter to be used with electrode data. The default is True (filter on)
    EEGdata_LPF_flag : BOOL, optional
        Turns on/off the low pass filter to be used with electrode data The default is False (filter off)

    Returns
    -------
    VEOG_data : NPARRAY
        VEOG data after high pass and/or low pass filtering
    HEOG_data : NPARRAY
        HEOG data after high pass and/or low pass filtering
    data : NPARRAY
        Electrode data after high pass and/or low pass filtering
    sample_rate : INT
        Sampling rate of the system. To be used in future functions

    """
    #### EOG data filtering
    
    ## High pass filter the data to remove slow drifts
    if EOG_HPF_flag == True:
        high_pass_filter = signal.butter(EOG_HPF_order, EOG_HPF_cutoff, 'hp', fs=sample_rate, output='sos') ##JORGE -- variable for HPF cutoff
        VEOG_data = signal.sosfilt(high_pass_filter, VEOG_data)
        HEOG_data = signal.sosfilt(high_pass_filter, HEOG_data)
    
    ## Low pass filter
    if EOG_LPF_flag == True:
        low_pass_filter = signal.butter(EOG_LPF_order, EOG_LPF_cutoff, 'lp', fs=sample_rate, output='sos') ##JORGE -- variable for LPF cutoff
        VEOG_data = signal.sosfilt(low_pass_filter,VEOG_data)
        HEOG_data = signal.sosfilt(low_pass_filter, HEOG_data)
    
    print('Loading data ...')
    print('  ')
    
    #### EEG electrode data filtering
    
    ## High pass filter the data to remove slow drifts
    if EEGdata_HPF_flag == True:
        high_pass_filter = signal.butter(EEGdata_HPF_order, EEGdata_HPF_cutoff, 'hp', fs=sample_rate, output='sos') ##JORGE -- variable for HPF cutoff
        data = signal.sosfilt(high_pass_filter, data)
    
    ## Low pass filter
    if EEGdata_LPF_flag == True:
        low_pass_filter = signal.butter(EEGdata_LPF_order, EEGdata_LPF_cutoff, 'lp', fs=sample_rate, output='sos') ##JORGE -- variable for LPF cutoff
        data = signal.sosfilt(low_pass_filter, data)
        
    return VEOG_data, HEOG_data, data, sample_rate

def FFT_Master_pipeline(data_matrix, time, length_of_segment, mode):
    """
    Recursive use of the FFT_pipeline function. Allows accessing a matrix containing all flicker freqs. for a single colour.
    Calculates and returns individual FFT for each flicker freq., in a matrix with the same size than the input

    Parameters
    ----------
    data_matrix : NPARRAY
        Data to calculate FFT on. This data can either be original or interpolated. As a convention, please stack the data 
        in an increasing flicker freq. order. Meaning, data_matrix[0,:] = 30 Hz --> data_matrix[5,:] = 59 Hz, for both modes.
        
        For induced mode 'i', data_matrix should have a shape of [flicker_periods, 300000]
        For evoked mode 'e', matrix with shape [flicker_periods, length of segment] = (6x5000) for 1 sec. segments. 
    time : INT
        Total time in which the stimulus was turned on. It must be calculated as (300000/length of segment)/sample rate.
        300,000 is the total amount of datapoints for each stimulus (as we know the experiment ran for a whole minute at a
        sampling rate of 5000 Hz). 
    length_of_segment : INT
        Size of desired segments, specified in SECONDS. A sampling rate of 5000 is defined inside this function, so
        length_of_segment should be = 1 if dealing with 1 sec. segments, and so on.
    mode : STR
        Depending on the data extraction algorithm
        If data is phase-locked = evoked, use mode 'e'
        If data is non-phase-locked = induced, use mode 'i'

    Returns
    -------
    fft_matrix : NPARRAY
        Returns the calculated FFTs in a matrix with the exact same shape and order as the input data_matrix. This matrix
        can be directly used with the FFT_plots function.

    """
    sample_rate = 5000 # definition of sample rate
    # creating a zero matrix of shape flicker_periods x length of segment
    fft_matrix = np.zeros([len(data_matrix),length_of_segment*sample_rate]) 
    for i in range(len(data_matrix)):
            # recursive call of FFT_pipeline, accessing each flicker freq. individually using loop index
            fft_matrix[i,:] = FFT_pipeline(data_matrix[i,:], time, length_of_segment, mode) 
    return fft_matrix

def FFT_pipeline(data, time, length_of_segment, mode):
    """
    Calculates individual FFTs (according to specified flicker freq.) according to total flicker time and length of segment,
    specified in seconds. This function starts with baseline correction of the segment, multiplication with a hanning window,
    and calculation of FFT using scipy.fftpack. The output will be given individually for each flicker freq. Please refer to
    FFT_Master_pipeline function to determine how to calculate several FFTs simultaneously and output in matrix form.
    
    BEWARE:
    For induced mode 'i', the non-phase-locked extraction of segments is done in this function. Please do not attempt to
    extract induced data before calling this function and using it as an input. Please refer to the data type for more info.

    Parameters
    ----------
    data : NPARRAY
        For induced mode 'i', data must be a vector of size 300000, containing 60 secs. recording of each color and flicker
        freq. This mode will separate the whole data into segments of specified length and calculate FFT on these segments
        individually.
        For evoked mode 'e', data is already separated into segments, so phase-locked extraction of data must be done
        outside this function. 
    time : INT
        Total time in which the stimulus was turned on. It must be calculated as (300000/length of segment)/sample rate.
        300,000 is the total amount of datapoints for each stimulus (as we know the experiment ran for a whole minute at a
        sampling rate of 5000 Hz). 
    length_of_segment : INT
        Size of desired segments, specified in SECONDS. A sampling rate of 5000 is defined inside this function, so
        length_of_segment should be = 1 if dealing with 1 sec. segments, and so on.
    mode : STR
        Depending on the data extraction algorithm
        If data is phase-locked = evoked, use mode 'e'
        If data is non-phase-locked = induced, use mode 'i'

    Returns
    -------
    matrix : NPARRAY
        Returns the individual FFT depending with the same length as the segments. If used with FFT_Master_pipeline, this 
        will produce an FFT matrix ready to be used with FFT_plots, independently of the mode used. At this point, the
        output matrix for both modes must have the same shape.

    """
    sample_rate = 5000 # definition of sample rate
    if mode == 'i': # non-phase-locked (induced) mode
        # creates zero matrix to alocate FFT data while separating into segments of the specified length
        matrix = np.zeros([time, length_of_segment*sample_rate])
        for i in range(time): # loop runs for the amount of time each condition runs. 60 times for 1 sec. segments
            # separates into segments of the specified length
            fft_segment = data[(length_of_segment*sample_rate*i):length_of_segment*sample_rate*(i+1)] 
            fft_segment = fft_segment - fft_segment.mean() # baseline correction
            fft_segment = fft_segment * np.hanning(len(fft_segment)) # multiply by hanning window
            fft_segment = np.abs(fftpack.fft(fft_segment)) # calculating FFT on segment
            matrix[i,:] = fft_segment # allocating FFT of segment. This separates all data into a matrix of individual segments
        # after obtaining all FFT segments, for each colour and flicker freq., we average across all segments
        matrix = matrix.mean(axis=0) # output of the master pipeline will be a matrix with shape [flicker_periods,length of segment]
    if mode == 'e': # phase-locked (evoked) mode
        # creates zero vector to alocate evoked FFT data for each color and flicker freq.
        matrix = np.zeros([length_of_segment*sample_rate])
        for i in range(time): # loop runs for the amount of time each condition runs. 60 times for 1 sec. segments
            # in this mode, data is already separated into segments
            fft_segment = data.mean(axis=0) # first, we avg across all segments
            fft_segment = fft_segment - fft_segment.mean() # baseline correction
            fft_segment = fft_segment * np.hanning(len(fft_segment)) # multiply by hanning window
            fft_segment = np.abs(fftpack.fft(fft_segment)) # calculating FFT on segment
            matrix = fft_segment # allocating FFT of whole condition
            # output of the master pipeline will be a matrix with shape [flicker_periods,length of segment]
        # plt.plot(fft_segment,'b') # use to plot individual FFT segments, otherwise comment
    return matrix

def FFT_plots(colour_ffts, colour_lp_ffts, flicker_periods, harmonics, subharmonics, length_of_segment, colour, peak_indices=np.array(['']), comparison=True):
    """
    Produces 3 figures for each colour, in the order: blue (1-3), red (4-6), green (7-9) and white (10-12), with 6 subplots 
    each, in the order: 30Hz --> 59Hz as stated in the FFT_Master_pipeline documentation. Please constrain yourself to 
    these conventions to guarantee correct functionality of this function. 
    
    Function also allows to simultaneously plot the calculated peaks obtained from SNR function, mainly to verify their
    correct calculation, but this is only optional.
    
    Function also allows to compare FFTs calculated from original data, and compare it with FFTs from interpolated (lp)
    data. If you dont want to compare plots, turn the comparison flag to False. This will only plot the FFTs from the
    original data. If, on the other hand, you wish to plot the interpolated data without comparison, please pass
    the lp_ffts matrix as the first argument, the original ffts matrix as the second argument, and turn the comparison
    flag to False.

    Parameters
    ----------
    colour_ffts : NPARRAY
        Matrix in the shape [flicker_periods,length of segment] containing the FFTs from ORIGINAL data for each colour. If
        you desire to plot only the interpolated (lp) data, please pass such a matrix as this argument instead.
    colour_lp_ffts : NPARRAY
        Matrix in the shape [flicker_periods,length of segment] containing the FFTs from INTERPOLATED (lp) data for each 
        colour. If you desire to plot only the interpolated data, please pass the original FFTs matrix as an argument here.
    flicker_periods : LIST, NPARRAY
        Vector containing the flicker periods for each condition, specified in ms. i.e. for a flicker freq. of 30 Hz. Flicker
        periods are calculated as round(1000/flicker freq.), these usually being (30, 36, 40, 45, 56 and 59 Hz).
    harmonics : INT
        Specifies the number of harmonics-1 to be plotted as vertical lines in 2nd plot of each colour. i.e. if harmonics = 3
        this will plot the fundamental freq. plus the first two harmonics, and zoom into the region of interest.
    subharmonics : INT
        Specified the number of subharmonics to be plotted as vertical lines in 3rd plot of each colour. i.e. if 
        subharmonics = 4, this will calculate the lines as 1/(2+subharmonics), meaning subharmonics in 1/2, 1/3, 1/4 and 1/5
        of the fundamental freq. Max. number of subharmonics now is limited to 7, although it can be increased by making
        a bigger subharm_colour vector
    length_of_segment : INT
        Duration of segment, specified in TIMEPOINTS. Please pass an argument in the form of length_of_segment (in sec.) x
        sample_rate
    colour : STR
        Determines the colour in which the main plot (colour_ffts) will be displayed, and allows correct looping between
        figures, complementary colours for comparison, and colour selection in peak_indices vector. A colour MUST be
        specified according to the plotted condition, i.e. blue_ffts will pass 'b' as an argument for this parameter.
    peak_indices : NPARRAY, optional
        A matrix containing peak_indices for all colours and flicker freqs, in the shape of: 
        [number of colours X flicker periods X number of desired peaks]. Please refer to SNR function documentation
        to see how this matrix is built.
        The default is np.array(['']). This will avoid plotting any peaks if no matrix is passed
    comparison : BOOL, optional
        Allows to plot the original and interpolated FFTs into the same subplot, in different colours. The default is True.
        Please set to False if you only want to plot either the original or the interpolated FFTs alone.

    Returns
    -------
    1st figure: Complete FFT with rescaled x-axis according to length of segment, for all flicker freqs.
    2nd figure: Zoom into region spanning from 0 to last harmonic of interest + some window, to allow visibility
    for all flicker freqs.
    3rd figure: Zoom into subharmonic region (spanning from 0 to fundamental + some window) for all flicker freqs.

    """
    sample_rate = 5000 # definition of sample rate
    timeaxis = np.arange(0,length_of_segment,sample_rate/length_of_segment) # rescales x axis when considering bigger segments

    ax_colour = 'k' # sets vertical line colour for harmonics
    subharm_colour = ['m','tab:orange','c','y', 'tab:pink', 'tab:brown','tab:purple'] # sets colours for vertical lines in subharmonics
    if colour == 'b': # for blue ffts
        figure_count = 1 # sets number of first figure, to avoid plot overriding
        comp_colour = 'tab:purple' # sets contrasting colour for comparison plot
        peak_colour = 0 # sets index for indication in peak_indices matrix
    if colour == 'r': # for red ffts
        figure_count = 4 # sets number of first figure, to avoid plot overriding
        comp_colour = 'brown' # sets contrasting colour for comparison plot
        peak_colour = 1 # sets index for indication in peak_indices matrix
    if colour == 'g': # for green ffts
        figure_count = 7 # sets number of first figure, to avoid plot overriding
        comp_colour = 'mediumseagreen' # sets contrasting colour for comparison plot
        peak_colour = 2 # sets index for indication in peak_indices matrix
    if colour == 'k': # for white ffts
        figure_count = 10 # sets number of first figure, to avoid plot overriding
        comp_colour = 'dimgray' # sets contrasting colour for comparison plot
        peak_colour = 3 # sets index for indication in peak_indices matrix
        ax_colour = 'r' # changes colour of vertical harmonic lines to red
        
    for plot_count in range(len(flicker_periods)): # for the total number of subplots in each figure, corresponding to the flicker periods
        # plotting full FFT
        fig = plt.figure(figure_count) # opens 1st figure
        ax = fig.add_subplot(2,3,plot_count+1) # obtain axis handle for each subplot
        # timeaxis -> rescaling of x axis, FFT vector for each colour, determined colour plot, label for legend
        ax.plot(timeaxis[0:length_of_segment],colour_ffts[plot_count,:], colour, label='Original')
        if comparison: # if comparison flag is true
            # also plots lp_ffts to compare with original FFTs. label for legend
            ax.plot(timeaxis[0:length_of_segment],colour_lp_ffts[plot_count,:], comp_colour, label='Interpolated')
        ax.set_title(str(round(1000/flicker_periods[plot_count])) + ' Hz') # sets subplot title to flicker freq.
        if plot_count+1 == 3: # to avoid putting the legend box on each subplot, we only put it for the 4th subplot
            ax.legend(bbox_to_anchor=(1.05, 0), loc='lower left') # on the left, which sends it to the right of the figure
        
        # plotting zoom into harmonics
        fig = plt.figure(figure_count+1) # opens 2nd figure
        ax = fig.add_subplot(2,3,plot_count+1) # obtain axis handle for each subplot
        # timeaxis -> rescaling of x axis, FFT vector for each colour, determined colour plot, label for legend
        ax.plot(timeaxis[0:length_of_segment],colour_ffts[plot_count,:], colour, label='Original')
        if str(peak_indices.dtype) != "<U1": # this basically means, if peak_indices exists (and is not the default str)
            # then plot the peaks as crosses. Please refer to plot_peaks documentation for more info
            plot_peaks(colour_ffts[plot_count,:],peak_indices[peak_colour,plot_count,:],length_of_segment/sample_rate,harmonics)
        if comparison: # if comparison flag is true
            # also plots lp_ffts to compare with original FFTs. label for legend
            ax.plot(timeaxis[0:length_of_segment],colour_lp_ffts[plot_count,:], comp_colour, label='Interpolated')
        ax.set_title(str(round(1000/flicker_periods[plot_count])) + ' Hz') # sets subplot title to flicker freq.
        for i in range (harmonics): # loop around the amount of desired harmonic lines
            # plots vertical lines starting from the fundamental (0+1) freq, and multiplying it recursively
            ax.axvline(x=round(1000/flicker_periods[plot_count])*(i+1), color=ax_colour, linestyle='--')
        ax.set_xlim([0, (1000*harmonics)/flicker_periods[plot_count] + 15]) # zooms into region with harmonics
        if plot_count+1 == 3: # to avoid putting the legend box on each subplot, we only put it for the 4th subplot
            ax.legend(bbox_to_anchor=(1.05, 0), loc='lower left') # on the left, which sends it to the right of the figure
           
        # plotting zoom into subharmonic region
        fig = plt.figure(figure_count+2) # opens 3rd figure
        ax = fig.add_subplot(2,3,plot_count+1) # obtain axis handle for each subplot
        # timeaxis -> rescaling of x axis, FFT vector for each colour, determined colour plot, label for legend
        ax.plot(timeaxis[0:length_of_segment],colour_ffts[plot_count,:], colour, label='Original')
        if comparison: # if comparison flag is true
            # also plots lp_ffts to compare with original FFTs. label for legend
            ax.plot(timeaxis[0:length_of_segment],colour_lp_ffts[plot_count,:], comp_colour, label='Interpolated')
        ax.set_title(str(round(1000/flicker_periods[plot_count])) + ' Hz') # sets subplot title to flicker freq.
        ax.axvline(x=round(1000/flicker_periods[plot_count]), color=ax_colour, linestyle='--') # prints vertical line for fundamental
        for i in range (subharmonics): # loop around the amount of desired subharmonic lines
            # subharmonics are calculated as a division of the fundamental, starting from 1/2 and increasing the denominator recursively
            ax.axvline(x=round(1000/flicker_periods[plot_count])*(1/(i+2)), color=subharm_colour[i], linestyle='--')
        ax.set_xlim([0, (1000/flicker_periods[plot_count]) + 5]) # zooms into region with subharmonics
        if plot_count+1 == 3: # to avoid putting the legend box on each subplot, we only put it for the 4th subplot
            ax.legend(bbox_to_anchor=(1.05, 0), loc='lower left') # on the left, which sends it to the right of the figure

def plot_peaks(colour_ffts,indices,length_of_segment,harmonics):
    """
    Plots the calculated peaks on the harmonic plots, for visual inspection of correctness, with an orange cross.
    If you do not want to plot these, do not pass the peak_indices as an argument of the FFT_plots function.

    Parameters
    ----------
    colour_ffts : NPARRAY
        Vector of all FFT data for a specific flicker freq. 
    indices : NPARRAY
        Matrix of shape [number of colours X flicker periods X number of desired freqs]. For a definition of freqs,
        please refer to SNR function documentation.
    length_of_segment : INT
        Definition of length of segment, specified in SECONDS. In FFT plots, the length of segment is defined in TIMEPOINTS,
        so length_of_segment/sample_rate must be passed as input for this to work. This will help rescaling the x-axis of
        peaks so they appear correctly where peaks are after axis rescalement.
    harmonics : INT
        Helps iterating over the number of obtained peaks, which is usually aligned with the number of harmonics -1 
        (fundamental). But more can be obtained, by tweaking the freqs variable in SNR function

    Returns
    -------
    Overlapping orange crosses at the calculated peaks of the FFTs with ORIGINAL data.

    """
    for i in range(harmonics): # iterates over all obtained points of interest (usually harmonics) to plot them all
        plt.plot(indices[i]/length_of_segment,colour_ffts[int(indices[i])],"x", color='tab:orange', markersize=12)

def SNR(all_ffts,flicker_period,length,window):
    """
    This function implements scipy.signal.find_peaks to obtain the peak values around the freq. of interest (which usually
    are: fundamental and harmonics, but could also work for subharmonics (just not yet)). Once the peaks have been obtained,
    their magnitudes and indices are returned. Additionally, the function calculates the signal-to-noise ratio (SNR) as
    the peak value at the freq. of interest, divided by the magnitude of surrounding freqs., determined by a window defined
    by the user when calling the function. Because we are refering to voltage signals, the SNR in decibel (dB) units is
    calculated as the base-10 log of such division, as stated in: SEARCH LITERATURE.
    
    DISCLAIMER: this function has some issues that are explained throughout the code. It works best when the peak is
    characteristic, but if it has low amplitude compared to surrounding noise, the peak will be miscalculated due to
    (most likely) the hardcoded windows. If you find an adequate way of exactly calculating the desired peaks, please feel
    free to change this function, and let me know!

    Parameters
    ----------
    all_ffts : NPARRAY
        A matrix containing all colours, flicker periods and FFT datapoints. It is a 3D matrix with shape:
        [number of colors X number of flicker periods X length of segment (in timepoints)]. i.e. for 1 sec segments, this
        will have a shape of [4,6,5000]. Please stack all colour FFTs in the order stated by convention: blue, red, green
        and white.
    flicker_periods : LIST, NPARRAY
        Vector containing the flicker periods for each condition, specified in ms. i.e. for a flicker freq. of 30 Hz. Flicker
        periods are calculated as round(1000/flicker freq.), these usually being (30, 36, 40, 45, 56 and 59 Hz).
    length : INT
        Length of segment, specified in SECONDS. This is used to rescale x-axis and let the peaks match the correct values
        when plotting segments bigger than 1 sec.
    window : INT
        Window of frequencies defined by the user. This will help obtaining a local avg of freqs. before and after the
        freq. of interest. If flicker freq. = 40 Hz with peak at 40 Hz, and a window of 5 (Hz), with a length of 1 sec., 
        this will take 5 freqs. before (35-39 Hz) and after (41-45) the peak, and avg. them to calculate the SNR of 
        that freq. of interest. Window is also rescaled when the length of segments is bigger.

    Returns
    -------
    The outputs follow the convention stated so far. Please constraint to this convention to ensure proper function.
    1st dimension: colours -- blue, red, green, white, in that order
    2nd dimension: flicker periods -- 30, 36, 40, 45, 56, 59 Hz, in that order
    3rd dimension: freqs. of interest -- fundamental, 1st harmonic, 2nd harmonic
    i.e. SNR[0,0,0] = SNR of blue, 30 Hz flicker freq, fundamental
    i.e. peaks[2,4,1] = peak magnitude of green, 56 Hz, 1st harmonic
    All outputs follow this shape and convention.
    peaks : NPARRAY
        Returns the magnitude of calculated peaks for each colour, flicker period and freq. of interest. Matrix is in the
        shape of: [number of colours X number of flicker periods X number of freqs]. This peaks are only used for SNR
        calculation, but could be used for other purposes, and thus is returned.
    peaks_index : NPARRAY
        Returns the index of found peaks for each colour, flicker period and freq. of interest. Matrix is in the shape of:
        [number of colours X number of flicker periods X number of freqs]. Indices can be used as an argument of the
        FFT_plots function to visually corroborate if the calculated peaks are correct according to the plot. These indices
        are rescaled depending on the length of segments
    SNR : NPARRAY
        Returns the calculated signal-to-noise ratio for each colour, flicker period and freq. of interest. Matrix is in
        the shape of: [number of colours X number of flicker periods X number of freqs]. SNR values are not used
        afterwards, but are the main output of this function, and can be inspected easily with variable explorer.

    """
    # freqs = number of freqs. in which we are interested, this is matched with the number of harmonics used more often (3), but
    # can be increased to include subharmonics, if required
    freqs = 3 # 3 = fundamental + 2 harmonics + 1 subharmonic (doesnt work but makes the calculation of others work for some reason)
    # zero matrices for outputs with the specified conventional shape
    peaks = np.zeros([len(all_ffts),len(flicker_period),freqs]) 
    peaks_index = np.zeros([len(all_ffts),len(flicker_period),freqs])
    SNR = np.zeros([len(all_ffts),len(flicker_period),freqs])
    for j in range(len(all_ffts)): # this will iterate through all colours
        for k in range(len(flicker_period)): # this will iterate through all flicker periods
            for i in range(freqs): # this will iterate through all freqs. of interest
                if i >= freqs: # when freqs - 1, this intends to calculate peak for at least the first subharmonic
                    # and assign it as the last index of the output matrix. BUT calculation of peak for subharmonic is not working
                    # and is also not relevant for the acquire data. Nonetheless, this part of the script is left, mainly to make
                    # everything else function properly, but also as future work to fix calculation of subharmonic peaks
                    # for reference on the general idea of how these lines work, please refer to the else condition
                    peaks[j,k,i] = np.max(all_ffts[j,k,round((length*1000/flicker_period[k])*0.5)-3:round((length*1000/flicker_period[k])*0.5)+3])
                    index = np.array(find_peaks(all_ffts[j,k,round((length*1000/flicker_period[k])*0.5)-10:round((length*1000/flicker_period[k])*0.5)+10],height=peaks[j,k,i]))
                    index = index[0]
                    peaks_index[j,k,i] = round((length*1000/flicker_period[k])*0.5)-index[0]+1
                else:
                    # this sets a hardcoded (=3) window around the freqs. of interest and obtains the max value
                    peaks[j,k,i] = np.max(all_ffts[j,k,round(length*1000/flicker_period[k])*(i+1)-3:round(length*1000/flicker_period[k])*(i+1)+3])
                    # this max value is used as a height requirement for the find_peaks function, meaning that it will save the
                    # index of the first peak value that is >= than the max. For it, another hardcoded (=10) window around this freq.
                    # of interest is set, avoiding the natural peaks observed in alpha band, for example. this can create a problem
                    # as it could detect a wrong peak which complies with the height but outside the user-defined window. I have
                    # tried to fix this but when the hardcoded window is smaller, the calculation of local avgs stops working
                    index = np.array(find_peaks(all_ffts[j,k,round(length*1000/flicker_period[k])*(i+1)-10:round(length*1000/flicker_period[k])*(i+1)+10],height=peaks[j,k,i]))
                    index = index[0] # find_peaks function returns more, with this line we only keep the indices
                    # because of the hardcoded +/- 10 window, we need to correct the index position by this same value
                    # we save this iterative index into our output matrix
                    peaks_index[j,k,i] = index[0]+round(length*1000/flicker_period[k])*(i+1)-10
                    # after determining where the peak is (which might be slightly different from exactly fundamental or harmonics
                    # due to rounding errors), we now use the user-defined windows to obtain timepoints before the peak freq.
                    local_avg1 = all_ffts[j,k,int(peaks_index[j,k,i])-window*length:int(peaks_index[j,k,i])]
                # same logic as before, but now it obtains the timepoints after the peak freq.
                local_avg2 = all_ffts[j,k,int(peaks_index[j,k,i])+1:int(peaks_index[j,k,i])+window*length+1]
                local_avg = np.zeros(2*window*length) # establishes a zeros vector for the whole window 
                local_avg[0:window*length] = local_avg1 # sets the values before the peak freq. to the first half of this vector
                local_avg[window*length:window*length*2] = local_avg2 # sets the values after the peka freq. to the second half of this vector
                SNR[j,k,i] = 10*np.log(peaks[j,k,i]/local_avg.mean()) # calculates SNR using the peak value and averaged surroundings
                # results in dB -- LOOK AT MANE2019
    return peaks, peaks_index, SNR

## STILL NOT WORKING.. PLEASE IGNORE UNTIL FURTHER NOTICE
def colour_flicker_sorting(flicker_period, data, count):
    if flicker_period*5 == 165:
        data_33 = np.copy(data)
        count += 1
        return count, data_33
    if flicker_period*5 == 140:
        data_28 = np.copy(data)
        count += 1
        return count, data_28
    if flicker_period*5 == 125:
        data_25 = np.copy(data)
        count += 1
        return count, data_25
    if flicker_period*5 == 110:
        data_22 = np.copy(data)
        count += 1
        return count, data_22
    if flicker_period*5 == 90:
        data_18 = np.copy(data)
        count += 1
        return count, data_18
    if flicker_period*5 == 85:
        data_17 = np.copy(data)
        count += 1
        return count, data_17
    if count == 0:
        return data_33, data_28, data_25, data_22, data_18, data_17