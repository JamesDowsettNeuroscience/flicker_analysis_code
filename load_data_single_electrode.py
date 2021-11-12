#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 17:03:15 2021

@author: James Dowsett
"""
import numpy as np
import mne
import matplotlib.pyplot as plt

subject = 2

file_name = 'S' + str(subject) + '_colour'

electrode_name = 'Pz'


# read the EEG data with the MNE function
raw = mne.io.read_raw_brainvision(file_name + '.vhdr')


# get the channel number corresponding to the electrode name
channel_names = raw.info.ch_names
electrode = channel_names.index(electrode_name)

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


