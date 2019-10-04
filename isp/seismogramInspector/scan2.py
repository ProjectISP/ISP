#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 20:05:06 2018

@author: robertocabieces
"""

##Scan with the obspy subprocess
import warnings
warnings.filterwarnings("ignore")

from pprint import pprint
from obspy.signal.trigger import coincidence_trigger
from obspy.core import read
from obspy import UTCDateTime as UDT, read, Trace, Stream
import numpy as np
from obspy.core import UTCDateTime
from obspy.signal.trigger import plot_trigger
from obspy.signal.trigger import classic_sta_lta
from obspy.signal.trigger import recursive_sta_lta
from os import listdir
from os.path import isfile, join
from scipy.signal import argrelextrema as pks


def scan(ficheros_procesar_path):
    #ficheros_procesar_path = "/Users/robertocabieces/Documents/obspy/Scan/sismogramas"
    #ficheros_procesados_path = "/Users/robertocabieces/Documents/obspy/local062/procesado"
    
    # Obtenemos el listado de ficheros a procesar
    obsfiles = [f for f in listdir(ficheros_procesar_path) if isfile(join(ficheros_procesar_path, f))]
    obsfiles.sort()
    
    ##creamos el stream con todo
    st = Stream()
    for filename in obsfiles:
        st += read(ficheros_procesar_path + '/'+ filename)
    
    
    
    
    st.filter('bandpass', freqmin=1, freqmax=10)
    #st.plot()
    #coincidence_trigger(trigger_type, thr_on, thr_off, stream, thr_coincidence_sum, 
    #trace_ids=None, max_trigger_length=1000000.0, delete_long_trigger=False, trigger_off_extension=0, details=False, event_templates={}, similarity_threshold=0.7, **options)[source]¶
    
    trig = coincidence_trigger("classicstalta", 10, 2, st, 4, sta=1, lta=40)
    pprint(trig)
    
    #caso de haber un triger
    if len(trig) > 0:
        t = trig[0]
        Time=t['time']
    else:
        Time=None
    return Time