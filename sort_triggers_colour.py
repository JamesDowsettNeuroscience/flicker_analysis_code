#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 12:25:37 2021

@author: James Dowsett
"""

import numpy as np



subject = 1

file_name = 'S' + str(subject) + '_colour'


### read triggers

f = open(file_name + '.vmrk') # open the .vmrk file, call it "f"

# use readline() to read the first line 
line = f.readline()

# empty lists to record trigger times for each colour
red_triggers = []
green_triggers = []
blue_triggers = []
white_triggers = []

# the names of the triggers, these can be seen in the .vmrk file
red_trigger_name = 'S  1'
green_trigger_name = 'S  2'
blue_trigger_name = 'S  3'
white_trigger_name = 'S  4'

flicker_trigger_name = 'S128'

colour_condition = 0

# use the read line to read further.
# If the file is not empty keep reading one line
# at a time, until the end
while line:
    
    # first check if the line contains a condition start trigger, indicating the colour 
    if red_trigger_name in line:
        colour_condition = 1
    elif green_trigger_name in line:
        colour_condition = 2
    elif blue_trigger_name in line:
        colour_condition = 3
    elif white_trigger_name in line:
        colour_condition = 4
    
    if flicker_trigger_name in line: # if the line contains a flicker trigger
        
        # get the trigger time from line
        start = line.find(flicker_trigger_name + ',') + len(flicker_trigger_name) + 1
        end = line.find(",1,")
        trigger_time = line[start:end]       
       
        # append the trigger time to the correct condition
        if colour_condition == 1:
            red_triggers.append(trigger_time)
        elif colour_condition == 2:
            green_triggers.append(trigger_time)
        elif colour_condition == 3:
            blue_triggers.append(trigger_time)        
        elif colour_condition == 4:
            white_triggers.append(trigger_time)             
    
    line = f.readline() # use realine() to read next line
    
f.close() # close the file

# convert to numpy arrays
all_red_triggers = np.array(red_triggers, dtype=np.int)

all_green_triggers = np.array(green_triggers, dtype=np.int)

all_blue_triggers = np.array(blue_triggers, dtype=np.int)

all_white_triggers = np.array(white_triggers, dtype=np.int)



## print out number of triggers, should be 15900 in total for each colour

print(str(len(all_red_triggers)) + ' red triggers') 
print(str(len(all_green_triggers)) + ' green triggers') 
print(str(len(all_blue_triggers)) + ' blue triggers') 
print(str(len(all_white_triggers)) + ' white triggers') 



# save files

print('Saving...')

np.save(file_name + '_all_red_triggers', all_red_triggers)
np.save(file_name + '_all_green_triggers', all_green_triggers)
np.save(file_name + '_all_blue_triggers', all_blue_triggers)
np.save(file_name + '_all_white_triggers', all_white_triggers)

print('Done')


