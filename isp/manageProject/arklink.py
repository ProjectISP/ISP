#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:45:24 2019

@author: robertocabieces
"""
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import os


def arklink(client1,net,sta,loc,channel,time1,time2,output,write=True):
    name=net+"."+sta+"."+loc+"."+channel
    try:
        print("Connecting to "+client1)
        client = Client(client1)
        
        t1 = UTCDateTime(time1)
        t2=  UTCDateTime(time2)
        t3=t2-t1
        print("Retrieving Waveforms")
        st = client.get_waveforms(net, sta, loc, channel, t1, t1+t3)
        print(st)
        if write:
            if os.path.isdir(output):
                print("writing mseed") 
                #st.write(output+"/"+name, format='MSEED')
                for tr in st: 
                    tr.write(output+"/"+tr.id + ".MSEED", format="MSEED") 
                #st.read(output+"/"+"*.*")
                #st.plot()
            else:
                print("Creating output directory")
                os.mkdir(output)
                print("writing mseed")
                #st.write(output+"/"+name, format='MSEED')
                for tr in st: 
                    tr.write(output+"/"+tr.id + ".MSEED", format="MSEED") 
                #st.read(output+"/"+"*.*")
                #st.plot()
                    
    except:
        print("An exception has happened")    
    

