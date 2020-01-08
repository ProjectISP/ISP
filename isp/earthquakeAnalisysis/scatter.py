#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 23:58:34 2019

@author: robertocabieces
"""
from obspy import read_events
from obspy.io.nlloc.util import read_nlloc_scatter
import numpy as np
import matplotlib.pyplot as plt
import os
import math as mt
path=os.getcwd()+"/location_output"+"/loc"
#data=read_nlloc_scatter("location.20150917.151207.grid0.loc.scat", coordinate_converter=None)
data=read_nlloc_scatter(path+"/last.scat")
L=len(data)
x=[]
y=[]
z=[]
pdf=[]
latOrig=33
lonOrig=-10
#x = (long - longOrig) * 111.111 * cos(lat_radians); 
#y = (lat - latOrig) * 111.111; 
#lat = latOrig + y / 111.111; 
#long = longOrig + x / (111.111 * cos(lat_radians)); 
for i in range(L):
    x.append(data[i][0])
    y.append(data[i][1])
    z.append(data[i][2])
    pdf.append(data[i][3])
x=np.array(x)
y=np.array(y)


##if thre trans is simple not NONE
conv=111.111*mt.cos(latOrig*180/mt.pi)
x=(x/conv)+lonOrig
y=(y/111.111)+latOrig
pdf=np.array(pdf)/np.max(pdf)

# definitions for the axes
left, width = 0.05, 0.65
bottom, height = 0.1, 0.65
spacing = 0.02


rect_scatter = [left, bottom, width, height]
rect_scatterlon = [left, bottom + height + spacing, width, 0.2]
rect_scatterlat  = [left + width + spacing, bottom, 0.2, height]

##Central Figure X,Y
fig=plt.figure(figsize=(8, 8))
#plt.figure(figsize=(8, 8))

ax_scatter = plt.axes(rect_scatter)
ax_scatter.tick_params(direction='in', top=True, right=True)

plt.scatter(x,y,s=10,c=pdf, alpha=0.5, marker=".",cmap=plt.cm.jet)

#ax_scatter.set_xlim((-11, -9))
#ax_scatter.set_ylim((35.5, 37))
plt.xlabel("Longitude (o)")
plt.ylabel("Latitude (o)")
##Figure top 
ax_scatx = plt.axes(rect_scatterlon)
ax_scatx.tick_params(direction='in', labelbottom=False)

plt.scatter(x,z,s=10,c=pdf, alpha=0.5, marker=".",cmap=plt.cm.jet)
#ax_scatx.set_xlim((-11, -9))
#ax_scatx.set_ylim((0, 60))
plt.ylabel("Depth (km)")
plt.gca().invert_yaxis()
ax_scatx = plt.axes(rect_scatterlat)
ax_scatx.tick_params(direction='in', labelleft=False)

###Figure right 
ax_scaty = plt.axes(rect_scatterlat)
ax_scaty.tick_params(direction='in')
#
plt.scatter(z,y,s=10,c=pdf, alpha=0.5, marker=".",cmap=plt.cm.jet)
#ax_scaty.set_xlim((0, 60))
#ax_scaty.set_ylim((35.5, 37))

ax_scaty = plt.axes(rect_scatterlat)
ax_scaty.tick_params(direction='in')
plt.xlabel("Depth (km)")
cax = plt.axes([0.93, 0.1, 0.02, 0.8])

plt.colorbar(cax=cax)
plt.show()
name="mapa2.png"
fig.savefig(name,dpi=800)