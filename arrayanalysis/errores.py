#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 18:39:40 2018

@author: robertocabieces
"""

###Errores
import math
import numpy as np
import matplotlib.pyplot as plt
import glob, os, shutil
pi = math.pi
def errores(path,i):
#    folder=path+"pictures"
#    path1=path
#    os.makedirs(folder)
    i=str(i)
    path = path+"/pow_map_"+i+".npz"
    #a=np.load("/Users/robertocabieces/Documents/obspy/local062/Slownessmap/pow_map_2904.npz")
    a= np.load(path)
    d = dict(zip((path), (a[k] for k in a)))
    
    fig, ax = plt.subplots()
    
    d.keys()
    Z = d["/"]
    L = len(Z)
    #A = np.zeros((L,L))
    A = list()
    S = list()
    #S = np.zeros((L,L))
    #Giramos 180 
    Z=np.fliplr(Z)
    Z=np.flipud(Z)
    #Siqueremos un txt de resultados
    #np.savetxt('z.csv', Z, delimiter=',')
    Lim=0.3/300 #definicion de cada pixel limite de slowness 
    #en un eje entre la mitad del numero pixels
    
    x=y=np.linspace(-0.3,0.3,601)
    X, Y = np.meshgrid(x, y)
    #plt.contourf(X, Y, array_aux, 10,cmap =plt.cm.bone)
    cs = ax.contourf(Y, X, Z,100,cmap=plt.cm.jet)
    plt.ylabel('Sy [s/km]')
    plt.xlabel('Sx [s/km]')
    cbar = fig.colorbar(cs)
    plt.show()
    #name=path[52:65]+".png"
    #fig.savefig(name)
    M=np.max(Z)-0.05*np.max(Z)
    for i in range(L):
        for j in range(L):
            if Z[i,j]<M:
                
                Z[i,j]=0
                
#    fig, ax = plt.subplots()
#    cs = ax.contourf(Y, X, Z,100,cmap=plt.cm.jet)
#    plt.ylabel('Sy [s/km]')
#    plt.xlabel('Sx [s/km]')
#    cbar = fig.colorbar(cs)
#    plt.show()
            
    for i in range(len(Z)):
        for j in range(len(Z)):
            if Z[i,j]>0:
                xx=i-1*(L/2)
                yy=j+(L/2)
                Angle=math.atan2(yy,xx)*180/pi
                Slowness=math.hypot(xx*Lim,yy*Lim)
                #A[i,j]=Angle*180/pi
                A.append(Angle)
                S.append(Slowness)
    A = np.array(A)
    S = np.array(S)         
    ErrorAngle = max(A)-min(A)
    ErrorSlowness = max(S)-min(S)
    #fig.savefig("S_Wavelet.png")
    
#    files = glob.iglob(os.path.join(path1, "*.png"))
#    for file in files:
#        if os.path.isfile(file):
#            shutil.copy2(file, folder)
#            shutil.move(folder,"*.png*")
#    
    return(ErrorAngle,ErrorSlowness)
   