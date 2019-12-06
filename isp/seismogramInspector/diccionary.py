#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 13:40:22 2019

@author: robertocabieces
"""
import warnings

import pandas as pd
from deprecated import deprecated


@deprecated(reason="This method is deprecated. See PickerManager")
def dictionary(x1,y1,sta,P_phase):
    warnings.warn('This method is deprecated.  See PickerManager', DeprecationWarning, stacklevel=2)
    #x1="2015-09-17T15:12:23.649431Z"
    #y1=-1.65e-07
    Date=x1[0:4]+x1[6:7]+x1[8:10]
    Hour_min=x1[11:13]+x1[14:16]
    Seconds=x1[18:20]
    Amplitude="%.2f" % y1
    
    
    ##Leer plantilla
    out=pd.read_csv("output.txt", sep=" ", index_col=None,nrows=None)
    try:
        out=out.drop(columns='Unnamed: 0')
    
        out2=out.append({'Station_name': sta,'Instrument':"?",'Component':"?",'P_phase_onset':"?",
     
            'P_phase_descriptor':P_phase,'First_Motion':"?",'Date':Date,'Hour_min':Hour_min,'Seconds':Seconds,
            'Err':"?",'Coda_duration':"?",'Amplitude':Amplitude,'Period':"?",'PriorWt':"?"},ignore_index=True)
    except:
    
        out2=out.append({'Station_name': sta,'Instrument':"?",'Component':"?",'P_phase_onset':"?",
     
                'P_phase_descriptor':P_phase,'First_Motion':"?",'Date':Date,'Hour_min':Hour_min,'Seconds':Seconds,
                'Err':"?",'Coda_duration':"?",'Amplitude':Amplitude,'Period':"?",'PriorWt':"?"},ignore_index=True)
    #Escribir datos nuevos
    out2.to_csv('output.txt',sep=" ",index = False)
    
    return
    
    