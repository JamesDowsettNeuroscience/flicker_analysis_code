#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 09:29:45 2021

@author: James Dowsett
"""


import mne

import numpy as np

subject = 1

file_name = 'pilot_' + str(subject)
        
print('Loading data: ' + file_name)

num_electrodes = 11

### first load the EEG data

# read the EEG data with the MNE function
raw = mne.io.read_raw_brainvision(file_name + '.vhdr', preload=True)


#### load data and save as a numpy file

## empty matrix to put interpolated data into
all_data = np.zeros((num_electrodes,len(raw))) 
        
for electrode in range(0, num_electrodes):
    
      print('Electrode: ' + str(electrode))
    
      electrode_data = np.array(raw[electrode,:], dtype=object)
    
      electrode_data = electrode_data[0,]
    
      electrode_data = electrode_data.flatten()
    
      all_data[electrode,:] = electrode_data
    
      #electrode_data = electrode_data - electrode_data.mean()
    

np.save(file_name + '_all_data', all_data)

print('Done')

print(str(np.round((len(electrode_data)/1000)/60,1)) + ' min of data')


### read triggers

f = open(file_name + '.vmrk') # open the .vmrk file, call it "f"

# use readline() to read the first line 
line = f.readline()

# empty lists to record trigger times 
triggers = []



trigger_name = 'M  1'

colour_condition = 0

# use the read line to read further.
# If the file is not empty keep reading one line
# at a time, until the end
while line:
    
    if trigger_name in line: # if the line contains a flicker trigger
        
        # get the trigger time from line
        start = line.find(trigger_name + ',') + len(trigger_name) + 1
        end = line.find(",1,")
        trigger_time = line[start:end]       
       
        # append the trigger time to all_triggers
       
        triggers.append(trigger_time)
              
    
    line = f.readline() # use realine() to read next line
    
f.close() # close the file

# convert to numpy arrays
all_triggers = np.array(triggers, dtype=np.int)



## print out number of triggers

print(str(len(all_triggers)) + ' triggers') 



# save files

print('Saving...')

np.save(file_name + '_all_triggers', all_triggers)

print('Done')