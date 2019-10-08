
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 12:07:15 2018

@author: robertocabieces
"""
import warnings

from isp.seismogramInspector.Auxiliary2 import MTspectrum

warnings.filterwarnings("ignore")
from obspy import read
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

win=150
tbp=3
ntapers=5
fmin=2
fsup=8
#ficheros_procesar_path="/Users/robertocabieces/Desktop/GUIPYTHON/ArrayProcessing/260/Velocity"
#ficheros_procesados=""
def MTspectrogram(ficheros_procesar_path,win,tbp,ntapers,fmin,fsup):

    obsfiles = [f for f in listdir(ficheros_procesar_path) if isfile(join(ficheros_procesar_path, f))]
    obsfiles.sort()
    nfilas=len(obsfiles)
    k=1
    fig = plt.figure()
    for f in obsfiles:
        if f != ".DS_Store":
            st1=read(ficheros_procesar_path+"/"+f)
            tr = st1[0]
            num=tr.stats.station
            net=tr.stats.network
            channel=tr.stats.channel
            dt = tr.stats.starttime
            tr.detrend()
            tr.taper
            npts = tr.stats.npts
            delta = tr.stats.delta
            t = np.linspace(0, (delta * npts), npts-win)
            mtspectrogram=MTspectrum(tr.data,win,delta,tbp,ntapers,fmin,fsup)
            M=np.max(mtspectrogram)
            mtspectrogram2=10*np.log(mtspectrogram/M)
            x, y = np.meshgrid(t,np.linspace(fmin, fsup, mtspectrogram2.shape[0]))

            # plt.ion()
            # subplot = plt.subplot(nfilas, 1, k)
            fig.add_subplot(nfilas, 1, k)

            cs = plt.contourf(x, y, mtspectrogram2, 100, cmap=plt.cm.jet)
            plt.title("Multi Taper Spectrogram"+num)
            plt.xlabel("Time after %s [s]" % tr.stats.starttime)
            plt.ylabel("Frequency [Hz]")
            plt.colorbar(cs)
            k += 1

    return fig

    # st=read(ficheros_procesar_path+"/"+"*.*")
    # #st.plot()
    # mp = MatplotlibFrame(st)
    # mp.show()
