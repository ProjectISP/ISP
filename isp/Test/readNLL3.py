#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 23:58:34 2019

@author: robertocabieces
"""
#from obspy import read_events
from obspy.io.nlloc.util import read_nlloc_scatter
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
#from plotmap1 import plotmap
data=read_nlloc_scatter("lecturas.20160115.013604.grid0.loc.scat", coordinate_converter=None)
L=len(data)
#xy=np.genfromtxt('lecturas.20160115.013604.grid0.loc.scat.lonlat.XY')
#zy=np.genfromtxt('lecturas.20160115.013604.grid0.loc.scat.ZY')
#x1=xy[:,0]
#min(x1)
#y1=xy[:,1]
#min(y1)
#z1=zy[:,0]
x=[]
y=[]
z=[]
pdf=[]
for i in range(L):
    x.append(data[i][0])
    y.append(data[i][1])
    z.append(data[i][2])
    pdf.append(data[i][3])


x=np.array(x)
y=np.array(y)
z=np.array(z)
pdf=np.array(pdf)/np.max(pdf)

f=111.111*np.cos(38.5*np.pi/180)

#diffx=min(x)-min(x1)
x=(x/f)-9
y=(y/(111.111))+38.5

# definitions for the axes
left, width = 0.06, 0.65
bottom, height = 0.1, 0.65
spacing = 0.02


rect_scatter = [left, bottom, width, height]
rect_scatterlon = [left, bottom + height + spacing, width, 0.2]
rect_scatterlat  = [left + width + spacing, bottom, 0.2, height]

##Central Figure X,Y
fig=plt.figure(figsize=(10, 8))
#plt.figure(figsize=(8, 8))

ax_scatter = plt.axes(rect_scatter)
ax_scatter.tick_params(direction='in', top=True, right=True,labelsize=10)

plt.scatter(x,y,s=10,c=pdf, alpha=0.5, marker=".",cmap=plt.cm.jet)

#ax_scatter.set_xlim((-11, -9))
#ax_scatter.set_ylim((35.5, 37))
plt.xlabel("Longitude",fontsize=10)
plt.ylabel("Latitude",fontsize=10)
##Figure top 
ax_scatx = plt.axes(rect_scatterlon)
ax_scatx.tick_params(direction='in', labelbottom=False,labelsize=10)

plt.scatter(x,z,s=10,c=pdf, alpha=0.5, marker=".",cmap=plt.cm.jet)
#ax_scatx.set_xlim((-11, -9))
#ax_scatx.set_ylim((0, 60))
plt.ylabel("Depth (km)",fontsize=10)
plt.gca().invert_yaxis()
ax_scatx = plt.axes(rect_scatterlat)
ax_scatx.tick_params(direction='in', labelleft=False,labelsize=10)

###Figure right 
ax_scaty = plt.axes(rect_scatterlat)
ax_scaty.tick_params(direction='in')
ax_scaty.tick_params(which='major',labelsize=10)
#
plt.scatter(z,y,s=10,c=pdf, alpha=0.5, marker=".",cmap=plt.cm.jet)
#plt.text(fontsize=18)
#ax_scaty.set_xlim((0, 60))
#ax_scaty.set_ylim((35.5, 37))

ax_scaty = plt.axes(rect_scatterlat)
ax_scaty.tick_params(direction='in',labelsize=10)
plt.xlabel("Depth (km)",fontsize=10)
cax = plt.axes([0.95, 0.1, 0.02, 0.8])

plt.colorbar(cax=cax)
plt.show()
#name="scatter.png"
#fig.savefig(name,dpi=800)

#fig=plt.figure()
#plotmap(y,x,pdf,-14,34,-5,38)
#plt.scatter(x,z,s=30,c=pdf, alpha=0.5, marker=".",cmap=plt.cm.jet)
