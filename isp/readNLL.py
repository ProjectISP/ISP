#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 12:28:09 2019

@author: robertocabieces
"""

from obspy import read_events
from obspy.io.nlloc.util import read_nlloc_scatter
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from plotmap1 import plotmap
cat = read_events("2015-09-17.hyp")
print(cat)
event = cat[0]
print(event)
origin = event.origins[0]
print(origin)
time=origin.time
data=read_nlloc_scatter("NLL.20171019091833.866997.4823.loc.scat", coordinate_converter=None)
L=len(data)
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
pdf=np.array(pdf)/np.max(pdf)


fig=plt.figure()
plt.subplot(221)
plotmap(y,x,pdf,-14,34,-5,38)
plt.subplot(223)
plt.scatter(x,z,s=30,c=pdf, alpha=0.5, marker=".",cmap=plt.cm.jet)
#cbar = plt.colorbar()
#cbar.set_label("PDF", labelpad=+1)
plt.title("Point observations")
plt.xlabel("Longitude (o)")
plt.ylabel("Depth (km)")
plt.gca().invert_yaxis()
plt.subplot(222)
plt.scatter(z,y,s=30,c=pdf, alpha=0.5, marker=".",cmap=plt.cm.jet)
plt.title("Point observations")
plt.xlabel("Depth (km)")
plt.ylabel("Latitude")
cax = plt.axes([0.95, 0.1, 0.02, 0.8])
plt.colorbar(cax=cax)
#cax.set_label("PDF", labelpad=+1)

plt.show()
name="mapa1.png"
fig.savefig(name,dpi=800)