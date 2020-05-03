#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 02:38:45 2019

@author: robertocabieces
"""

from mcovvar import *
SP=[]
file='mcovvar.txt'
A=np.genfromtxt(file,dtype='str')
Size=int(np.size(A)/12)

for i in range(Size):
    [d,F,P]=computeOriginErrors(i,file)
    
    SolucionE=str(d['depth_errors'])+"     "+str(P['azimuth_max_horizontal_uncertainty'])+"     "+str(P['min_horizontal_uncertainty']) +"     "+str(P['max_horizontal_uncertainty'])
    SP.append(SolucionE + "\n")


file = open ('StatisticsE.txt', 'w') 
file.writelines(SP)
file.close()
