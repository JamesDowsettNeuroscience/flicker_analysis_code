#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 10:58:30 2023

@author: James Dowsett
"""

### analysis pipeline for SIT-EEG experiments



from flicker_analysis_package import functions
import numpy as np
import matplotlib.pyplot as plt

import scipy.signal

path = '/home/james/Active_projects/SIT_EEG/SIT_pilot/'


file_names = ('AKG_overear', 'ear_buds')

electrode_names = ['C5', 'C3', 'Cz', 'C4', 'C6', 'Pz', 'Oz', 'REF2']


sample_rate = 1000

period = 250 # 4 Hz sound

plt.figure()

# for file_name in file_names:

file_name = file_names[0]
 
 ### load triggers
 
walking_triggers_file_name = file_name + '_all_triggers.npy'

all_triggers = np.load(path + walking_triggers_file_name)


## sort triggers

diff_triggers = np.diff(all_triggers)

# empty lists to put the triggers into for each condition
search_triggers_list = []
target_triggers_list = []

for trig_count in range(0,len(diff_triggers)):

    # if trigger is approx. 250 ms before the next trigger then this is a search trigger
    if diff_triggers[trig_count] <= 252 and diff_triggers[trig_count] >= 248:
        search_triggers_list.append(all_triggers[trig_count])
         
     # if trigger is a target trigger (target triggers come in pairs, only use the first which has approx. 121 ms time difference)
    if diff_triggers[trig_count] <= 123 and diff_triggers[trig_count] >= 117:
         
         ## check trigger is not part of demonstration sounds which come at the begining of each trial
         
         next_6_triggers = diff_triggers[trig_count:trig_count+6]
         
         if max(next_6_triggers) < 132 and min(next_6_triggers) > 117: # must be 6 consecutive triggers in correct time range
         
             target_triggers_list.append(all_triggers[trig_count])
         
 
# convert to numpy arrays
search_triggers = np.array(search_triggers_list, dtype=int)
target_triggers = np.array(target_triggers_list, dtype=int)

print('\n' + file_name + '  ' + str(len(search_triggers)) + ' search triggers')
print('\n' + file_name + '  ' + str(len(target_triggers)) + ' target triggers')  

## make triggere time series for plotting

# all_triggers_time_series = np.zeros([max(all_triggers)+1, ])
# search_triggers_time_series = np.zeros([max(all_triggers)+1, ])
# target_triggers_time_series = np.zeros([max(all_triggers)+1, ])


# for trigger in all_triggers:
#     all_triggers_time_series[trigger] = 10

# for trigger in search_triggers:
#     search_triggers_time_series[trigger] = 5

# for trigger in target_triggers:
#     target_triggers_time_series[trigger] = 5

# plt.plot(all_triggers_time_series)
# plt.plot(search_triggers_time_series)
# plt.plot(target_triggers_time_series)


## get REF2 data
data_file_name = file_name + '_chan_7_data.npy'

REF2_data = np.load(path + data_file_name)         

 
 
 ### loop for each electrode
 
#plt.figure()
# plt.suptitle(file_name)
 
plot_count = 0

 #for electrode in (2,5,6):
electrode = 2

data_file_name = file_name + '_chan_' + str(electrode) +  '_data.npy'

data = np.load(path + data_file_name)         

# re-reference
data = data - (REF2_data/2)

 ## filter
cutoff = 40 # cutoff in Hz
nyquist = 1000/2
Wn = cutoff/nyquist
b, a = scipy.signal.butter(3, Wn, 'low')
data = scipy.signal.filtfilt(b, a, data)

 
 
# make SSVEP
search_SSVEP = functions.make_SSVEPs(data, search_triggers, period) #

target_SSVEP = functions.make_SSVEPs(data, target_triggers, period) # 




## baseline correct
search_SSVEP = search_SSVEP - search_SSVEP.mean()
target_SSVEP = target_SSVEP - target_SSVEP.mean()





## calculate std. deviation values

search_seg_matrix = np.zeros([len(search_triggers),period]) 
trig_count = 0
for trig in search_triggers:
    search_seg_matrix[trig_count,:] = data[trig:trig+period]
    trig_count += 1

search_std = np.zeros([period,])
for t in range(0,period):
    search_std[t] = np.std(search_seg_matrix[:,t])

target_seg_matrix = np.zeros([len(target_triggers),period]) 
trig_count = 0
for trig in target_triggers:
    target_seg_matrix[trig_count,:] = data[trig:trig+period]
    trig_count += 1

target_std = np.zeros([period,])
for t in range(0,period):
    target_std[t] = np.std(target_seg_matrix[:,t])


# calculate std error

search_std_error = search_std/np.sqrt(len(search_triggers))
target_std_error = target_std/np.sqrt(len(target_triggers))

search_std_error = search_std_error /20
target_std_error  = target_std_error /20

 ## convert to micro- volts
 
search_SSVEP = search_SSVEP * 1000000
target_SSVEP = target_SSVEP * 1000000

search_std_error = search_std_error * 1000000
target_std_error = target_std_error * 1000000

plot_count += 1
#plt.subplot(1,3,plot_count)

plt.title(electrode_names[electrode])
 
# if file_name == 'AKG_overear':
#      plt.plot(search_SSVEP, 'b', label = 'Search Overear')
#      plt.plot(target_SSVEP, 'r', label = 'Target Overear')
# elif file_name == 'ear_buds':
#      plt.plot(search_SSVEP, 'c', label = 'Search Ear buds')
#      plt.plot(target_SSVEP, 'm', label = 'Target Ear buds')        
     
plt.plot(search_SSVEP, 'b', label = 'Search')
plt.plot(target_SSVEP, 'r', label = 'Target')
    
 
 # plt.legend()

# plt.ylim([-2.5, 4.5])
# plt.ylim([-2.5, 2.5])


plt.xlabel('Time (ms)')

plt.ylabel('Voltage (\u03BCV)')

plt.legend()




from matplotlib import pyplot as plt


x = np.linspace(0, period, period)

plt.fill_between(x, search_SSVEP-search_std_error, search_SSVEP+search_std_error,
                 alpha=0.2, edgecolor='b', facecolor='c',
                 linewidth=1, linestyle='solid', antialiased=True)

plt.fill_between(x, target_SSVEP-target_std_error, target_SSVEP+target_std_error,
                 alpha=0.2, edgecolor='r', facecolor='m',
                 linewidth=1, linestyle='solid', antialiased=True)




plt.show()



