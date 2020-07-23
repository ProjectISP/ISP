# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 17:54:07 2020

@author: olivar
"""

import obspy

import pickle
import matplotlib.pyplot as plt

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

def map_data(path, quick=True):
    
    data_map = {}
    
    for top_dir, sub_dir, files in os.walk(path):
        for file in files:
            full_path_to_file = os.path.join(top_dir, file)
            
            if quick:
                year = int(file.split('.')[0])
                jday = int(file.split('.')[1])
                stnm = file.split('.')[3]
                chnm = file.split('.')[5]
            else:
                header = #read header
                year = header.stats.year
                jday
                stnm
                chnm

            if os.stat(full_path_to_file).st_size > 0:
                data_map.setdefault(stnm, {})
                data_map[stnm].setdefault(year, {})
                data_map[stnm][year].setdefault(jday, {})
                data_map[stnm][year][jday].setdefault(chnm, full_path_to_file)
            
    return data_map

#data_map = pickle.load(open("C:/Users/olivar/Work/data_map.pickle", "rb"))
data_map = map_data("path", quick=False)

fig = plt.figure(figsize=(7.27, 10.69))
fig.subplots(nrows=1, ncols=1)
ax = fig.axes[0]
fig.subplots_adjust(left=0.125, bottom=0.02, right=0.95, top=0.98)


labels = []
labels_ypos = []

yorder = 0
label_ypos = yorder + 0.5
for stnm in list(sorted(data_map.keys())):

    labels.append(stnm)
    labels_ypos.append(label_ypos)
    
    years = sorted(list(data_map[stnm].keys()))
    first_year = min(years)
    last_year = max(years)
    
    rectangles = []
    accumulated_days = int((obspy.UTCDateTime(year=first_year, julday=1) - obspy.UTCDateTime(year=2014, julday=1))/86400)
    
    for year in years:
        
        days = sorted(list(data_map[stnm][year].keys()))
        first_day = min(days)
        last_day = max(days)
    
        if year == first_year:
            max_days = int((obspy.UTCDateTime(year=year+1, julday=1) - obspy.UTCDateTime(year=year, julday=first_day))/86400)
        elif year > first_year and year < last_year:
            max_days = int((obspy.UTCDateTime(year=year+1, julday=1) - obspy.UTCDateTime(year=year, julday=1))/86400)
        elif year == last_year:
            max_days = int((obspy.UTCDateTime(year=year, julday=last_day) - obspy.UTCDateTime(year=year, julday=1))/86400)
        
        rectangle_width = 0
        rectangle_color = "green"
        rectangle_left_corner = (first_day + accumulated_days, yorder)
        
        for i in range(max_days):
            jday = first_day + i
            
            if jday in days:
                if rectangle_color == "red":
                    # Create a red rectangle ...
                    rectangles.append(Rectangle(rectangle_left_corner, width=rectangle_width + 1, height=1, facecolor=rectangle_color))
                    # ... and start the next one (green)
                    rectangle_width = 0
                    rectangle_color = "green"
                    rectangle_left_corner = (jday + accumulated_days, yorder)
                else:
                    rectangle_width += 1
            else:
                if rectangle_color == "green":
                    # Create a green rectangle ...
                    rectangles.append(Rectangle(rectangle_left_corner, width=rectangle_width + 1, height=1, facecolor=rectangle_color))
                    # ... and start the next one (red)
                    rectangle_width = 0
                    rectangle_color = "red"
                    rectangle_left_corner = (jday + accumulated_days, yorder)
                else:
                    rectangle_width += 1
            
        # Create the last rectangle
        rectangles.append(Rectangle(rectangle_left_corner, width=rectangle_width + 1, height=1, facecolor=rectangle_color))
        
        accumulated_days += jday
    
    for r in rectangles:
        ax.add_patch(r)

    yorder += 1
    label_ypos += 1

ax.set_yticks(labels_ypos)
ax.set_yticklabels(labels)
ax.set_xlim(0, 2400)


"""fig = plt.figure()
ax = fig.add_subplot(111)

pc = PatchCollection([Rectangle((0, 0), width=8, height=1)], facecolor="green")

ax.add_collection(pc)"""