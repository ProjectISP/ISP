#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 12:07:15 2018

@author: robertocabieces
"""
import warnings
warnings.filterwarnings("ignore")
from obspy import UTCDateTime as UDT, read, Trace, Stream
from obspy.core import UTCDateTime
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import obspy
from obspy.imaging.cm import obspy_sequential
#from obspy.signal.tf_misfit import cwt
from ccwt import *
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema as pks


def wavelet(ficheros_procesar_path,ficheros_procesados,fmin,fmax,wmin,wmax,tt):
    figsize = (10, 8)
    obsfiles = [f for f in listdir(ficheros_procesar_path) if isfile(join(ficheros_procesar_path, f))]
    obsfiles.sort()
    nfilas=len(obsfiles)
    k=1    
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
          t = np.linspace(0, delta * npts, npts-1)
          
          nf = 40
          scalogram = ccwt(tr.data, 1/delta, fmin, fmax, wmin, wmax,tt, nf)
          scalogram1 = np.abs(scalogram)**2
          maxscalogram1 = np.max(scalogram1)
          scalogram2 = 10*(np.log10(scalogram1/maxscalogram1)) #pasar visualizar pasar db
          
          plt.ion()
          
          fig=plt.subplot(nfilas,1,k)
          
          x, y = np.meshgrid(t,np.logspace(np.log10(fmin), np.log10(fmax), scalogram.shape[0]))
          cs = fig.contourf(x, y, scalogram2,nf,cmap=plt.cm.jet)
          #plt.legend("CWT "+ num)
          plt.xlabel("Time after %s [s]" % tr.stats.starttime,fontsize=6)
          plt.xticks(rotation=40,fontsize=6)
          plt.ylabel("Frequency [Hz]",fontsize=6)
          plt.yticks(rotation=40,fontsize=6)
          cbar=plt.colorbar(cs)
          cbar.ax.tick_params(which='both',labelsize=6 )
          
          #plt.colorbar(mappable, ax=self.axes, orientation='horizontal',
                         #cax=self.cbar_ax
          #cbar.ax.tick_params(labelsize=10)
          
          plt.show()
          name = "CWTOBS"+".png" 
          plt.savefig(name,dpi=800)
          #pp=np.abs(scalogram)
          pp=scalogram1
          ppp=np.log10(pp)
          pdif=np.diff(ppp)
          size = pdif.shape
          Mat = []
          for j in range(npts):
              if j < size[1] :
                  fila = np.mean(pdif[:,j])
                  Mat.append(fila)
              else:
                  print("")
          
          data=np.array(Mat) 
          
          
          start_time =dt
          sampling_rate = 1/delta
          data_size = len(data)
          
          #thresh = 0.0025
          #idx = pks(data,np.greater, order=3) 
          #jdx = np.where((data[idx]> thresh)) 
            
          #kdx = idx[0][jdx[0]]
          #kdx = np.array(kdx) #Tiempos
          
          #Deltatime= (kdx[1]-kdx[0])/(sampling_rate)
          #Deltatime = np.array(Deltatime)
          
          stats = {'network': net, 'station': num, 'location': '',
        	  'channel': channel, 'npts': len(data), 'sampling_rate': sampling_rate,
        	  'mseed': {'dataquality': 'D'}}
        
          stats['starttime'] = dt
        
          st2 = Stream([Trace(data=data, header=stats)])

          st2 = st2.filter('lowpass', freq=0.1, corners=3, zerophase=True)

          st2.taper(max_percentage=0.1)

          st2.write(ficheros_procesados+"/"+f, format='MSEED')
          k=k+1
    #Plotear Sismograma
    path1=ficheros_procesar_path+"/"+"*.*"
    st0=read(path1)
    st0.filter('highpass', freq=0.5, corners=4, zerophase=True)
    st0.plot(outfile="Wavelet_Analysis",size=(500,700),handle=True)
    #Plotear SCTFs
    path2=ficheros_procesados+"/"+"*.*"
    st3=read(path2)
    st3.plot(outfile="CF_Analysis",color='red',size=(500,700),handle=True)
        