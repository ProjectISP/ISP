#from PyQt4 import Qt, QtGui, uic, QtCore, QFont
from PyQt4 import *
from PyQt4 import Qt, QtGui, uic, QtCore
from PyQt4.QtCore import pyqtSlot, pyqtSignal, QThread, QEventLoop, QObject
from ccwt import *
from MTspectrogram import *
from Auxiliary import *
from Auxiliary2 import *
import matplotlib.pyplot as plt
import time
import numpy as np
from obspy.core import UTCDateTime
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
from diccionary import *
import os
from scan2 import scan
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from recursos_1_rc import *
from traveltimes import *
from readNLLevent import getNLLinfo
from scipy.signal import hilbert
from polarity2 import *
ventana_principal = "MainWindow.ui" 
MainWindow, _ = uic.loadUiType(ventana_principal)

class MyMainWindow(QtGui.QMainWindow, MainWindow):

    def __init__(self, parent):
        QtGui.QMainWindow.__init__(self)
        MainWindow.__init__(self)
        self.setupUi(self)
        # App
        self.app = parent
        ##Inicialización Parametros Picking
        
        
        self.Net1.setText("WM")
        self.Sta1.setText("OBS01")
        self.Loc1.setText("")
        self.Channel1.setText("SHZ")
        
        self.Dateplot1.setText("2015-09-17T15:11:00.00")
        self.secplot1.setText("240")
        self.Dateplot2.setText("2015-09-17T15:11:00.00")
        self.secplot2.setText("240")
        
        self.n1.setValue(99)
        self.n2.setValue(99)
   
                
       #Inicializacion Plot
       
       
        self.Pathwaveforms.setText("/Users/robertocabieces/Desktop/GUIPYTHON/ArrayProcessing/260/Velocity")
        # Inicializacion de elementos de la interfaz Seismogram Inspector.
        self.pathseismogram.setText("/Users/robertocabieces/Desktop/GUIPYTHON/ArrayProcessing/260/RAW")
        self.pathdataless.setText("/Users/robertocabieces/Desktop/GUIPYTHON/ArrayProcessing/260/dataless")
        self.pathoutput.setText("/Users/robertocabieces/Desktop/GUIPYTHON/ArrayProcessing/260/Velocity")
        #self.Date.setText("2015-09-17T15:11:00.00")
        #self.sec.setText("240")
        self.resultados.setText("Seismogram Info")
        self.Sta.setValue(1)
        self.Lta.setValue(40)
        self.Fmin.setValue(2)
        self.Fmax.setValue(10)
        
        self.FminPol.setValue(0.1)
        self.FmaxPol.setValue(0.5)
        self.TWPol.setValue(10)
        self.sec.setValue(240)
        
        self.comboPick.addItems(["Phase","P","Pn","Pg","S","Sn","Sg","Lg"])
        self.comboFilter.addItems(["Filter","bandpass","bandstop","lowpass","highpass"])
                
        #Conections
                
        self.Plot_Seismogram.clicked.connect(self.__test4__)
        self.rotate.clicked.connect(self.rot)
        
        self.STA_LTA.clicked.connect(self.__test5__)
        self.Envelope.clicked.connect(self.__test6__)
        self.Spectrum.clicked.connect(self.__test7__)
        self.Deconvolve.clicked.connect(self.__test8__)
        self.Spectrogram.clicked.connect(self.__test9__)
        self.PRS1.clicked.connect(self.__test10__)
        self.PlotSeismogram.clicked.connect(self.__test11__)
        self.PlotSeismogram2.clicked.connect(self.__test12__)        
        self.DP1.clicked.connect(self.__test13__)
        self.DP2.clicked.connect(self.__test14__)
        self.getevent.triggered.connect(self.__test15__)
        self.scan.clicked.connect(self.__test16__)
        self.PAnalysis.clicked.connect(self.PolsAnl)
        
        
        ##Funcion del menubar para crear un projecto
        
        self.LoadPath0.clicked.connect(self.load0)
        self.LoadPath1.clicked.connect(self.load1)
        self.LoadPath2.clicked.connect(self.load2)
        self.LoadPathout.clicked.connect(self.loadout)
        
        self.LoadFile.clicked.connect(self.loadFile)
        self.geteventinfo.clicked.connect(self.eventinfopick)
        self.actionMagnitude_Coherence.triggered.connect(self.Coherence)
       
        self.actionCross_Correlation.triggered.connect(self.CrossCorr)
        ##Funcion del menubar tools
         
        self.PlotSingleRecord.triggered.connect(self.plotsinglerecord)
        #self.PlotAllRecords.triggered.connect(self.PlotAll)
       
        try:
            os.remove("output.txt")
        except:
             pass
         
        lst={'Station_name': ["Sta"],'Instrument':["Instr"],'Component':["Comp"],'P_phase_onset':["Pho"],
     
            'P_phase_descriptor':["P_phase"],'First_Motion':["First_Motion"],'Date':["Date"],'Hour_min':["Hour"],'Seconds':["Seconds"],
            'Err':["Err"],'Coda_duration':["Coda"],'Amplitude':["Ampl"],'Period':["Period"],'PriorWt':["PriorWt"]}
        
        df=pd.DataFrame(lst,columns=['Station_name','Instrument','Component','P_phase_onset','P_phase_descriptor','First_Motion','Date','Hour_min'
                            'Seconds','Err','Coda_duration','Amplitude','Period','PriorWt'])
        
        df.to_csv('output.txt',sep=" ",index = True)
    
    def loadFile(self):
        startfolder=os.getcwd
        startfolder=str(startfolder)
        seismogram_file1 = QtGui.QFileDialog.getOpenFileName(self, "Open file",startfolder)
        self.Pathwaveforms.setText(seismogram_file1)
        
        
    def load0(self):
        startfolder=os.getcwd
        startfolder=str(startfolder)
        my_dir = QtGui.QFileDialog.getExistingDirectory(self,"Open a folder",startfolder,QtGui.QFileDialog.ShowDirsOnly)
        self.Pathwaveforms.setText(my_dir)
        
    def load1(self):
        startfolder=os.getcwd
        startfolder=str(startfolder)
        my_dir = QtGui.QFileDialog.getExistingDirectory(self,"Open a folder",startfolder,QtGui.QFileDialog.ShowDirsOnly)
        self.pathseismogram.setText(my_dir)
        
    def load2(self):
        startfolder=os.getcwd
        startfolder=str(startfolder)
        my_dir = QtGui.QFileDialog.getExistingDirectory(self,"Open a folder",startfolder,QtGui.QFileDialog.ShowDirsOnly)
        self.pathdataless.setText(my_dir)
        
    def loadout(self):
        startfolder=os.getcwd
        startfolder=str(startfolder)
        my_dir = QtGui.QFileDialog.getExistingDirectory(self,"Open a folder",startfolder,QtGui.QFileDialog.ShowDirsOnly)
        self.pathoutput.setText(my_dir)
        
        
    def plotsinglerecord(self):
        startfolder1=os.getcwd()
        #seismogram_file1 = QtGui.QFileDialog.getOpenFileName(self, "Open file",startfolder1)
        #self.Pathwaveforms.setText(seismogram_file1)
        seismogram_file=self.Pathwaveforms.text()
        #singleplot("/Users/robertocabieces/Documents/ISP/260/Velocity/WM.OBS01..SHZ.D.2015.260")
        singleplot(seismogram_file)
    
    def PlotAll(self):        
        #startfolder2=os.getcwd()
        #folder = QtGui.QFileDialog.getExistingDirectory(self,"Open a folder",startfolder2,QtGui.QFileDialog.ShowDirsOnly)
        folder=self.Pathwaveforms.text()
        folder=str(folder)
        allplot(folder)
        
    def eventinfopick(self):
        path=os.getcwd()+"/seismogramInspector"
        pathfile = QtGui.QFileDialog.getOpenFileName(self, 'Open file',path)
        time,latitude,longitude,depth=getNLLinfo(pathfile)
        self.LAT.setValue(latitude)
        self.LON.setValue(longitude)
        self.DEPTH.setValue(depth)
        self.eventtime.setText(str(time))
    

    @pyqtSlot()
    def __test16__(self):
        Path=self.pathseismogram.text()
        scan1(Path)
        
        
    @pyqtSlot()
    def __test13__(self):
        from obspy import read
        
        Net=self.Net1.text()
        Sta=self.Sta1.text()
        Loc=self.Loc1.text()
        Channel=self.Channel1.text()
        
        f1=self.Freq1p1.value()
        f2=self.Freq2p1.value()
        
               
        Path1=self.Pathwaveforms.text()
        Path=Path1+"/"+"*.*"
        st=read(Path)
        n1=self.n1.value()
                
        if n1 > 98:
            ## format WM.OBS01..SHZ.D.2015.260##
           Path=Path1+"/"+Net+"."+Sta+"."+Loc+"."+Channel+"*"
           st=read(Path)
           if f1 and f2 >0:
            if f1<f2:
                st.filter('bandpass', freqmin=f1, freqmax=f2, corners=4, zerophase=True)
                st.plot(type='dayplot')
           else:                               
             st.plot(type='dayplot')
                
    @pyqtSlot()
    def __test14__(self):
        from obspy import read
        Net=self.Net2.text()
        Sta=self.Sta2.text()
        Loc=self.Loc2.text()
        Channel=self.Channel2.text()
        
        f1=self.Freq1p2.value()
        f2=self.Freq2p2.value()
        
               
        Path1=self.Pathwaveforms.text()
        Path=Path1+"/"+"*.*"
        st=read(Path)
        n1=self.n1.value()
                
        if n1 > 98:
            ## format WM.OBS01..SHZ.D.2015.260##
           Path=Path1+"/"+Net+"."+Sta+"."+Loc+"."+Channel+"*"
           st=read(Path)
           if f1 and f2 >0:
            if f1<f2:
                st.filter('bandpass', freqmin=f1, freqmax=f2, corners=4, zerophase=True)
                st.plot(type='dayplot')
           else:                               
             st.plot(type='dayplot')

    def __test15__(self):
        Path1=self.Pathwaveforms.text()
        Time=scan(Path1)
        self.pickinfo.setText(str(Time))
    
    
    
    @pyqtSlot()
    def __test11__(self):
        global x_corr
        import numpy as np
        import matplotlib.pyplot as plt
        from obspy import read
        from obspy.core import UTCDateTime
        bbox = dict(boxstyle="round",fc="white")
        bbox1 = dict(boxstyle="round", fc="white")
        self.PlotSeismogram.setEnabled(False)
        self.grafico1.canvas.ax1.clear()
        self.grafico1.canvas.ax2.clear()
        self.grafico1.canvas.ax3.clear()
        Filter=self.comboFilter.currentText()       
        Net=self.Net1.text()
        Sta=self.Sta1.text()
        Loc=self.Loc1.text()
        Channel=self.Channel1.text()
        
        f1=self.Freq1p1.value()
        f2=self.Freq2p1.value()
        
        t1=self.Dateplot1.text()
        t2=float(self.secplot1.text())
        
        t1=UTCDateTime(t1)
        t2=t1+t2
 
        T1=t1
        
        
        Path1=self.Pathwaveforms.text()
        Path=Path1+"/"+"*.*"
        
        st=read(Path,starttime=t1,endtime=t2)
        st.detrend()
        st.taper(max_percentage=0.05)
        L1=len(st)-1
        #L2=self.n1.value()
        if f1 and f2 >0 and Filter == 'bandpass' or Filter == 'bandstop':
            if f1<f2:
                st.filter(Filter, freqmin=f1, freqmax=f2, corners=4, zerophase=True)
                st.detrend()
                
        elif Filter == 'highpass':
             print(Filter)
             f1=self.Freq1p1.value()
             tr=st[L1]
             f2=(tr.stats.sampling_rate)/2
             st.filter(Filter,freq=f1,corners=4, zerophase=True)
             st.detrend()
             print("Filter Done")
             
        elif Filter == 'lowpass':
             print(Filter)
             f1=0
             f2=self.Freq2p1.value()                          
             st.filter(Filter,freq=f2,corners=4, zerophase=True)
             st.detrend()
             print("Filter Done")
        else:
            f1=0
            tr=st[L1]
            f2=(tr.stats.sampling_rate)/2
        
        n1=self.n1.value()

        
        fmin=f1
        fsup=f2
        print(st)
        
        if n1 > 98:
            ## format WM.OBS01..SHZ.D.2015.260##
           Path=Path1+"/"+Net+"."+Sta+"."+Loc+"."+Channel+"*"
           st=read(Path,starttime=t1,endtime=t2)
           if f1 and f2 >0 and Filter == 'bandpass' or Filter == 'bandstop':
            if f1<f2:
                st.filter(Filter, freqmin=f1, freqmax=f2, corners=4, zerophase=True)
                st.detrend()
                       
           elif Filter == 'highpass':
             print(Filter)
             f1=self.Freq1p1.value()
             print(str(f1))
             f2=(tr.stats.sampling_rate)/2
             st.filter(Filter,freq=f1,corners=4, zerophase=True)
             st.detrend()
             print("Filter Done")
             
           elif Filter == 'lowpass':
             print(Filter)
             f1=0
             f2=self.Freq2p1.value()                          
             st.filter(Filter,freq=f2,corners=4, zerophase=True)
             st.detrend()
             print("Filter Done")
           else:
            f1=0
            tr=st[0]
            f2=(tr.stats.sampling_rate)/2
                
           n1=0
        
        
        fmin=f1
        fsup=f2
        
        tr=st[n1]
        starttime1=tr.stats.starttime
        sta=tr.stats.station
        starttime=str(starttime1)
        tr.detrend()
        y=tr.data
        npts=len(tr.data)
        samprate=tr.stats.sampling_rate
        delta=1/samprate
        Fs=1/samprate
        t1 = np.arange(0, npts / samprate, 1 / samprate)
        
        OFFSET=max(y)
                       
        if self.MTPSPEC.isChecked():
            
            win=int((self.WLen1.value())*samprate)
            tbp=self.TWMT.value()
            ntapers=self.NTAPERS.value()
            
           
                    
            t2 = np.linspace(0, (delta * npts), npts-win)
        
            mtspectrogram=MTspectrum(tr.data,win,delta,tbp,ntapers,fmin,fsup)          
            M=np.max(mtspectrogram)
            mtspectrogram2=10*np.log(mtspectrogram/M) 
            minMTS=np.min(mtspectrogram2)
            maxMTS=np.max(mtspectrogram2)
            
            X, Y = np.meshgrid(t2,np.linspace(fmin, fsup, mtspectrogram2.shape[0]))
            self.grafico1.canvas.ax2.clear()
            self.grafico1.canvas.ax2.contourf(X,Y,mtspectrogram2,100,cmap=plt.cm.jet)
            ColorbarBase(self.grafico1.canvas.cax, cmap=plt.cm.jet,norm=Normalize(vmin=minMTS, vmax=maxMTS))
            self.grafico1.canvas.cax.set_title("Power [dB]") 
        else:
            pass
            #self.grafico1.canvas.ax2.clear()
         
            
        if self.CWT.isChecked():
         if fmin==0:
           fmin=1/(self.WLen2.value())  
      
         nf = 40
         tt=int((self.WLen2.value())*samprate)
         wmin=self.W1.value()
         wmax=self.W2.value()
         t = np.linspace(0, delta * npts, npts-1)
         scalogram1 = ccwt(tr.data, samprate, fmin, fsup, wmin, wmax,tt, nf) 
         scalogram1 = np.abs(scalogram1)**2
          
         maxscalogram1 = np.max(scalogram1)
          
         scalogram2 = 10*(np.log10(scalogram1/maxscalogram1))  
          
         maxCWT=np.max(scalogram2)
          
         minCWT=np.min(scalogram2)
          
          
         self.grafico1.canvas.ax2.clear()
         X, Y = np.meshgrid(t,np.linspace(fmin, fsup, scalogram2.shape[0]))
         self.grafico1.canvas.ax2.contourf(X,Y,scalogram2,100,cmap=plt.cm.jet)
          
         ColorbarBase(self.grafico1.canvas.cax, cmap=plt.cm.jet,norm=Normalize(vmin=minCWT, vmax=maxCWT))
         self.grafico1.canvas.cax.set_title("Power [dB]") 
        else:
          pass
        #self.grafico1.canvas.ax2.clear()
        
                
        def onclick(event):
                       
            if event.dblclick:
                
                if event.button == 1:
                    Phase=self.comboPick.currentText()
                                            
                    x1, y1 = event.xdata, event.ydata
                    
                    self.grafico1.canvas.ax1.annotate(Phase, xy=(x1, 0), xytext=(x1,-1*OFFSET),bbox=bbox,arrowprops=dict(facecolor='red', shrink=0.05))
                    self.grafico1.canvas.ax1.plot(x1,0,'|',markersize=100,color='red')                                        
                    self.grafico1.canvas.draw()
                    x1=str(starttime1+x1)
                        
                   #Introducir funcion que crea fichero de datos NLLOC
                    dictionary(x1,y1,sta,Phase)
                    self.pickinfo.setText(Phase+" "+x1)
                    print(Phase, x1)
                    
                if event.button == 2 and self.actionPlot_Spectrum.isChecked() == True:
                    #timewindow=float(self.secplot1.text())
                    timewindow=int(10/delta)
                    x2 = event.xdata
                    x2=int(x2/delta)
                    #tt=np.arange([x2*delta,timewindow*delta,delta])
                    yy=y[x2:x2+timewindow]
                    yy=yy-np.mean(yy)
                    plt.plot(yy)
                    spectrumelement(yy,delta,sta)
                
                if event.button ==3:
                    self.grafico1.canvas.mpl_disconnect(cid)
                    self.PlotSeismogram.setEnabled(True)
                    print("Picking closed")
#        

        
        self.grafico1.canvas.ax1.clear()
        self.grafico1.canvas.ax1.plot(t1,y,label=sta,linewidth=0.5,color='k')
        self.grafico1.canvas.ax1.set_xlim(t1[0],t1[len(t1)-1])                
        self.grafico1.canvas.ax1.legend()
        self.grafico1.canvas.ax1.set_ylabel("Amplitude")
        self.grafico1.canvas.ax1.set_title("Seismogram vs Time-Frequency Analysis")
        #self.grafico1.canvas.ax2.set_title("Mtp.Spectrogram")
        self.grafico1.canvas.ax2.set_ylabel("Frequency [Hz]")
        self.grafico1.canvas.ax2.set_xlabel( "Time [s] after  "+starttime)
                                       
        cid=self.grafico1.canvas.mpl_connect('button_press_event', onclick)
        
        
        if self.actionPlotEnvelope.isChecked() == True:
            analytic_sygnal=hilbert(y)
            envelope=np.abs(analytic_sygnal)
            self.grafico1.canvas.ax1.plot(t1,envelope,linewidth=0.5,color='sandybrown')
            
        if self.actionSTALTA.isChecked() == True:
            stalta=classic_sta_lta_py(y, 1, 40)            
            self.grafico1.canvas.ax3.plot(t1,stalta,linewidth=0.5,color='grey',alpha=0.5)
            self.grafico1.canvas.ax3.set_ylim(-15,15)
            self.grafico1.canvas.ax3.set_ylabel("STA/LTA")
            
        if  self.actionEntropy.isChecked() == True:
            win=2**8
            dt=delta
            data=y
            t_entropy = np.linspace(0, (delta * npts), npts-win)+(win*dt/2)

            Entropy=Entropydetect(data,win,dt)            
            self.grafico1.canvas.ax3.plot(t_entropy,Entropy[0],linewidth=0.5,color='blue',alpha=0.5)
            self.grafico1.canvas.ax3.set_ylim(-1,1)
            self.grafico1.canvas.ax3.set_ylabel("Spectral Entropy")
        
        if self.ArrivalTimes.isChecked()==True:
            print("Plotting IASP91 Theretical Arrival Times")
            t3=T1-UTCDateTime(self.eventtime.text())
            t3=float(t3)
            eventlat=self.LAT.value()
            eventlon=self.LON.value()
            eventdepth=self.DEPTH.value()
            [Phases,Time]=arrivals2(eventlat,eventlon,eventdepth,sta) 
            print(Phases)
            print(Time)
            for i in range(len(Phases)):
                self.grafico1.canvas.ax1.annotate(Phases[i], xy=(Time[i]-t3,0), xytext=(Time[i]-t3, -1*OFFSET),bbox=bbox1,arrowprops=dict(facecolor='green', shrink=0.05))
                self.grafico1.canvas.ax1.plot(Time[i]-t3,0,'|',markersize=100,linestyle='dashed',color='green')
        self.grafico1.canvas.draw()
        x_corr=y
        
        return x_corr



    @pyqtSlot()
    def __test12__(self):
        global y_corr
        import numpy as np
        import matplotlib.pyplot as plt
        from obspy import read
        from obspy.core import UTCDateTime
        bbox = dict(boxstyle="round",fc="white")
        bbox1 = dict(boxstyle="round", fc="white")
        self.PlotSeismogram2.setEnabled(False)
        self.grafico2.canvas.ax1.clear()
        self.grafico2.canvas.ax2.clear()
        Filter=self.comboFilter.currentText()
        Net=self.Net2.text()
        Sta=self.Sta2.text()
        Loc=self.Loc2.text()
        Channel=self.Channel2.text()
        
        f1=self.Freq1p2.value()
        f2=self.Freq2p2.value()
        
        t1=self.Dateplot2.text()
        t2=float(self.secplot2.text())
        
        t1=UTCDateTime(t1)
        T1=t1
        t2=t1+t2        
        
        Path1=self.Pathwaveforms.text()
        Path=Path1+"/"+"*.*"
        
        st=read(Path,starttime=t1,endtime=t2)
        st.detrend()
        st.taper(max_percentage=0.05)
        L1=len(st)-1
        if f1 and f2 >0 and Filter == 'bandpass' or Filter == 'bandstop':
            if f1<f2:
                st.filter(Filter, freqmin=f1, freqmax=f2, corners=4, zerophase=True)
                st.detrend()
                
        elif Filter == 'highpass':
             print(Filter)
             f1=self.Freq1p2.value()
             tr=st[L1]
             f2=(tr.stats.sampling_rate)/2
             st.filter(Filter,freq=f1,corners=4, zerophase=True)
             st.detrend()
             print("Filter Done")
             
        elif Filter == 'lowpass':
             print(Filter)
             f1=0
             f2=self.Freq2p2.value()                          
             st.filter(Filter,freq=f2,corners=4, zerophase=True)
             st.detrend()
             print("Filter Done")
        else:
            f1=0
            tr=st[L1]
            f2=(tr.stats.sampling_rate)/2
            
        n2=self.n2.value()
    
        
        fmin=f1
        fsup=f2
        print(st)
        
        if n2 > 98:
            ## format WM.OBS01..SHZ.D.2015.260##
           Path=Path1+"/"+Net+"."+Sta+"."+Loc+"."+Channel+"*"
           st=read(Path,starttime=t1,endtime=t2)
           if f1 and f2 >0 and Filter == 'bandpass' or Filter == 'bandstop':
            if f1<f2:
                st.filter(Filter, freqmin=f1, freqmax=f2, corners=4, zerophase=True)
                st.detrend()
                       
           elif Filter == 'highpass':
             print(Filter)
             f1=self.Freq1p2.value()
             print(str(f1))
             f2=(tr.stats.sampling_rate)/2
             st.filter(Filter,freq=f1,corners=4, zerophase=True)
             st.detrend()
             print("Filter Done")
             
           elif Filter == 'lowpass':
             print(Filter)
             f1=0
             f2=self.Freq2p2.value()                          
             st.filter(Filter,freq=f2,corners=4, zerophase=True)
             st.detrend()
             print("Filter Done")
           else:
            f1=0
            tr=st[0]
            f2=(tr.stats.sampling_rate)/2
           n2=0
        
        fmin=f1
        fsup=f2
        print(st)   
        tr=st[n2]
        starttime1=tr.stats.starttime
        sta=tr.stats.station
        starttime=str(starttime1)
        tr.detrend()
        y=tr.data
        npts=len(tr.data)
        samprate=tr.stats.sampling_rate
        delta=1/samprate
        Fs=1/samprate
        t1 = np.arange(0, npts / samprate, 1 / samprate)
        
        OFFSET=max(y)
        
        if self.MTPSPEC.isChecked():
            
            win=int((self.WLen1.value())*samprate)
            tbp=self.TWMT.value()
            ntapers=self.NTAPERS.value()
            
           
                    
            t2 = np.linspace(0, (delta * npts), npts-win)
        
            mtspectrogram=MTspectrum(tr.data,win,delta,tbp,ntapers,fmin,fsup)          
            M=np.max(mtspectrogram)
            mtspectrogram2=10*np.log(mtspectrogram/M) 
            minMTS=np.min(mtspectrogram2)
            maxMTS=np.max(mtspectrogram2)
            
            X, Y = np.meshgrid(t2,np.linspace(fmin, fsup, mtspectrogram2.shape[0]))
            self.grafico2.canvas.ax2.clear()
            self.grafico2.canvas.ax2.contourf(X,Y,mtspectrogram2,100,cmap=plt.cm.jet)
            ColorbarBase(self.grafico2.canvas.cax, cmap=plt.cm.jet,norm=Normalize(vmin=minMTS, vmax=maxMTS))
            self.grafico2.canvas.cax.set_title("Power [dB]") 
        else:
            pass
            #self.grafico1.canvas.ax2.clear()
         
            
        if self.CWT.isChecked(): 
          if fmin==0:
               fmin=1/(self.WLen2.value())
          nf = 40
          tt=int((self.WLen2.value())*samprate)
          wmin=self.W1.value()
          wmax=self.W2.value()
          t = np.linspace(0, delta * npts, npts-1)
          scalogram1 = ccwt(tr.data, samprate, fmin, fsup, wmin, wmax,tt, nf) 
          scalogram1 = np.abs(scalogram1)**2
          
          maxscalogram1 = np.max(scalogram1)
          
          scalogram2 = 10*(np.log10(scalogram1/maxscalogram1))  
          
          maxCWT=np.max(scalogram2)
          
          minCWT=np.min(scalogram2)
          
          
          self.grafico2.canvas.ax2.clear()
          X, Y = np.meshgrid(t,np.linspace(fmin, fsup, scalogram2.shape[0]))
          self.grafico2.canvas.ax2.contourf(X,Y,scalogram2,100,cmap=plt.cm.jet)
          
          ColorbarBase(self.grafico2.canvas.cax, cmap=plt.cm.jet,norm=Normalize(vmin=minCWT, vmax=maxCWT))
          self.grafico2.canvas.cax.set_title("Power [dB]") 
        else:
            pass
            #self.grafico1.canvas.ax2.clear()
        
                
        def onclick(event):
                       
            if event.dblclick:
                
                if event.button == 1:
                    Phase=self.comboPick.currentText()
                       
    ##                        
                    x1, y1 = event.xdata, event.ydata
    ##                            
                    self.grafico2.canvas.ax1.annotate(Phase, xy=(x1, 0), xytext=(x1,-1*OFFSET),bbox=bbox,arrowprops=dict(facecolor='red', shrink=0.05))
                    self.grafico2.canvas.ax1.plot(x1,0,'|',markersize=100,color='red')             
                    self.grafico2.canvas.draw()
                    x1=str(starttime1+x1)
                        
                   #Introducir funcion que crea fichero de datos NLLOC
                    dictionary(x1,y1,sta,Phase)
                    self.pickinfo.setText(Phase+" "+x1)
                    print(Phase, x1)
                    
                if event.button == 2 and self.actionPlot_Spectrum.isChecked() == True:
                    #timewindow=float(self.secplot1.text())
                    timewindow=int(10/delta)
                    x2 = event.xdata
                    x2=int(x2/delta)
                    #tt=np.arange([x2*delta,timewindow*delta,delta])
                    yy=y[x2:x2+timewindow]
                    yy=yy-np.mean(yy)
                    plt.plot(yy)
                    spectrumelement(yy,delta,sta)                    
                if event.button ==3:
                    self.grafico2.canvas.mpl_disconnect(cid)
                    self.PlotSeismogram2.setEnabled(True)
                    print("Picking closed")
    #        
        
        self.grafico2.canvas.ax1.clear()
        self.grafico2.canvas.ax1.plot(t1,y,label=sta,linewidth=0.5,color='k')
        self.grafico2.canvas.ax1.set_xlim(t1[0],t1[len(t1)-1])                
        self.grafico2.canvas.ax1.legend()
        self.grafico2.canvas.ax1.set_ylabel("Amplitude")
        self.grafico2.canvas.ax1.set_title("Seismogram vs Time-Frequency Analysis")
        #self.grafico1.canvas.ax2.set_title("Mtp.Spectrogram")
        self.grafico2.canvas.ax2.set_ylabel("Frequency [Hz]")
        self.grafico2.canvas.ax2.set_xlabel( "Time [s] after  "+starttime)
                                       
        cid=self.grafico2.canvas.mpl_connect('button_press_event', onclick)
        
        if self.actionPlotEnvelope.isChecked() == True:
            analytic_sygnal=hilbert(y)
            envelope=np.abs(analytic_sygnal)
            self.grafico2.canvas.ax1.plot(t1,envelope,linewidth=0.5,color='sandybrown')
            
        if self.actionSTALTA.isChecked() == True:
            stalta=classic_sta_lta_py(y, 1, 40)            
            self.grafico2.canvas.ax3.plot(t1,stalta,linewidth=0.5,color='darkblue',alpha=0.3)
            self.grafico2.canvas.ax3.set_ylim(-20,20)
            self.grafico2.canvas.ax3.set_ylabel("STA/LTA")
            
        if  self.actionEntropy.isChecked() == True:
            win=2**8
            dt=delta
            data=y
            t_entropy = np.linspace(0, (delta * npts), npts-win)+(win*dt/2)
            Entropy=Entropydetect(data,win,dt)            
            self.grafico2.canvas.ax3.plot(t_entropy,Entropy[0],linewidth=0.5,color='blue',alpha=0.5)
            self.grafico2.canvas.ax3.set_ylim(-1,1)
            self.grafico2.canvas.ax3.set_ylabel("Spectral Entropy")
            
        if self.ArrivalTimes.isChecked()==True:
            print("Plotting IASP91 Theretical Arrival Times")
            t3=T1-UTCDateTime(self.eventtime.text())
            t3=float(t3)
            eventlat=self.LAT.value()
            eventlon=self.LON.value()
            eventdepth=self.DEPTH.value()
            [Phases,Time]=arrivals2(eventlat,eventlon,eventdepth,sta) 
            print(Phases)
            print(Time)
            for i in range(len(Phases)):
                self.grafico2.canvas.ax1.annotate(Phases[i], xy=(Time[i]-t3,0), xytext=(Time[i]-t3, -1*OFFSET),bbox=bbox1,arrowprops=dict(facecolor='green', shrink=0.05))
                self.grafico2.canvas.ax1.plot(Time[i]-t3,0,'|',markersize=100,linestyle='dashed',color='green')
                    
        self.grafico2.canvas.draw()
        y_corr=y
        
        return y_corr
    
    def CrossCorr(self):
                          
        if len(x_corr) != 0 and len(y_corr) != 0: 
            print("Computing Cross Correlation") 
            fs=50            
            #cc=np.correlate(x_corr, y_corr, "full")            
            cc = correlate_maxlag(x_corr, y_corr,len(x_corr))
            t = np.arange(-len(cc)//2,len(cc)//2) 
            t=t/fs
            #t = np.arange(0, len(cc) / fs, 1 / fs)            
            #fig = plt.figure()
            plt.plot(t,cc, label='Cross Correlation')
            plt.legend()
            plt.title("Normalized Cross Correlation")
            plt.show()
            
    def  Coherence(self):
        if len(x_corr) != 0 and len(y_corr) != 0:
            print("Computing Coherence between Signal 1 and Signal 2")
            fs=50
            nfft=2**8
            cohe(x_corr,y_corr,fs,nfft)
            
    @pyqtSlot()
    def __test4__(self):
        
        Path=self.pathseismogram.text()
        if self.Full.isChecked():
           st=read(Path+"/"+"*.*")           
           #allplot(Path)
           S=get_info2(Path)
           st.plot()
           self.resultados.setText(S)           
        else:                      
           time=self.Date.dateTime().toString("yyyy-MM-dd hh:mm:ss")
           Date=time[0:10]+"T"+time[11:19]        
           dt=self.sec.value() 
           dt=float(dt)                            
           t1=UTCDateTime(Date)           
           st=read(Path+"/"+"*.*",starttime=t1,endtime=t1+dt)
           S=get_info(Path,t1,dt)
           self.resultados.setText(S)
           st.plot()

           
    @pyqtSlot()
    def __test5__(self):
            
           Path=self.pathseismogram.text()
           Date=self.Date.text()
           t=float(self.sec.text())
           f_min=self.Fmin.value()
           f_max=self.Fmax.value()
           sta=self.Sta.value()
           lta=self.Lta.value()
           Sta_Lta(Path,Date,t,f_min,f_max,sta,lta)
           
    @pyqtSlot()
    def __test6__(self):
           ###envelope
           Path=self.pathseismogram.text()
           Date=self.Date.text()
           t=float(self.sec.text())
           f_min=self.Fmin.value()
           f_max=self.Fmax.value()
           get_envelope(Path,Date,t,f_min,f_max)
    
    @pyqtSlot()
    def __test7__(self):
           Path=self.pathseismogram.text()
           Date=self.Date.text()
           t=float(self.sec.text())
           spectrum(Path,Date,t)
           
    @pyqtSlot()
    def __test8__(self):
        if self.units_ACC.isChecked():
           physical_unit = self.units_ACC.text()
        if self.units_DIS.isChecked():
           physical_unit = self.units_DIS.text()
        if self.units_VEL.isChecked():
           physical_unit = self.units_VEL.text()
           
        print(physical_unit)
        Path=self.pathseismogram.text()
        Dataless=self.pathdataless.text()
        out=self.pathoutput.text()
        deconv(Path, Dataless, out, physical_unit)
        
    @pyqtSlot()        
    def __test9__(self):
        #MTspectrogram(ficheros_procesar_path,win,tbp,ntapers,fmin,fsup)
        lf = self.LFs.value()
        hf = self.HFs.value()
        tbp=3        
        ntapers=5
        win=150
        Path=self.pathseismogram.text()
        MTspectrogram(Path,win,tbp,ntapers,lf,hf)
        
    @pyqtSlot()        
    def __test10__(self):       
        #plrsection(path,path2,lat,lon)
        lat=float(self.lat.text())
        lon=float(self.lon.text())
        Path=self.pathseismogram.text()
        filePath = QtGui.QFileDialog.getOpenFileName(self, 'Open file', '/Users/robertocabieces/Desktop/GUIPYTHON/ArrayProcessing')
        plrsection(Path,filePath,lat,lon)
        
    def rot(self):
        ###rotation of seismograms
        directory=os.getcwd()
        output=self.pathoutput.text()
        Date=self.Date.text()
        t=float(self.sec.text())
        tr1 = QtGui.QFileDialog.getOpenFileName(self, 'Open file', directory)
        tr2= QtGui.QFileDialog.getOpenFileName(self, 'Open file', directory)
        deg1=self.deg1.value()
        deg2=self.deg2.value()
        rotate(tr1,tr2,deg1,deg2,Date,t,output,save=True)
        
    def PolsAnl(self):
        directory=os.getcwd()
        time=self.Date.dateTime().toString("yyyy-MM-dd hh:mm:ss")
        time1=time[0:10]+"T"+time[11:19]     
        #time2=UTCDateTime(time1)
        dt=self.sec.value()
        #time3=time2+dt
        trz = QtGui.QFileDialog.getOpenFileName(self, 'Open file', directory)
        trn= QtGui.QFileDialog.getOpenFileName(self, 'Open file', directory)
        tre= QtGui.QFileDialog.getOpenFileName(self, 'Open file', directory)
        f1=self.FminPol.value()
        f2=self.FmaxPol.value()        
        tw=self.TWPol.value()
        AP(time1,dt,trz,trn,tre,f1,f2,tw)
            