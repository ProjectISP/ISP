#from PyQt4 import Qt, QtGui, uic, QtCore, QFont
from PyQt4 import *
from PyQt4 import Qt, QtGui, uic, QtCore
from PyQt4.QtCore import pyqtSlot, pyqtSignal, QThread, QEventLoop, QObject
from Respuestaarray import arf
from wavelet import *
from ccwt import *
from FK import *
from Auxiliary import *
#from Auxiliary2 import *
import matplotlib.pyplot as plt
import time
import numpy as np
from matplotlib.figure import Figure
import pandas as pd
from diccionary import *
from FKcoherence import *
import os
from matplotlib.colorbar import ColorbarBase
from pathlib import Path
masterpath=Path(__file__).resolve().parent
masterpath=str(masterpath)

# Definimos la localizacion de los ficheros ui (las interfaces graficas)
ventana_principal = "MainWindow.ui"  # Enter file here. Carga todo lo que creemos con QtDesigner MainWindow.ui
# Cargamos los ficheros ui.
MainWindow, _ = uic.loadUiType(ventana_principal)


class MyMainWindow(QtGui.QMainWindow, MainWindow):

    # Definicion de signals
    #resultChanged = pyqtSignal(float, name="resultChanged")

    def __init__(self, parent):
        QtGui.QMainWindow.__init__(self)
        MainWindow.__init__(self)
        self.setupUi(self)
        # App
        self.app = parent

        # Inicializacion de elementos de la interfaz ARF.
        self.LF.setValue(0.25)
        self.HF.setValue(0.3)
        self.Smin.setValue(0)
        self.Smax.setValue(0.3)
        
        
        # Inicializacion de elementos de la interfaz Wavelet.
        self.LF1.setValue(2)
        self.HF1.setValue(12)
        self.NC1.setValue(6)
        self.NC2.setValue(8) 
        self.WD.setValue(2) 
        self.line1.setText("/Users/robertocabieces/Desktop/GUIPYTHON/ArrayProcessing/260/Velocity")
        self.line2.setText("/Users/robertocabieces/Desktop/GUIPYTHON/ArrayProcessing/260/SCTF")
                
        # Inicializacion de elementos de la interfaz FK.
        self.LF2.setValue(0.1)
        self.HF2.setValue(0.15)
        self.MaxSlow.setValue(0.3)
        self.SlowRes.setValue(0.01)
        self.Wfrac.setValue(0.01)
        self.TW.setValue(24)
        self.path_FK.setText("/Users/robertocabieces/Desktop/GUIPYTHON/ArrayProcessing/260/SCTF")
        self.path_coords.setText("/Users/robertocabieces/Documents/obs_array")
        self.DateFK.setText("2015-09-17T15:11:00.00")
        self.secFK.setText("200")
        #Conections
        self.pb_test.clicked.connect(self.__test1__)
        self.check1.clicked.connect(self.__test2__)
        self.RunFK.clicked.connect(self.__test3__)        
        
        ##Funcion del menubar para crear un projecto
        #self.LoadPath1.clicked.connect(self.load)
        ##
        
#    def load(self):
#        startfolder=os.getcwd
#        startfolder=str(startfolder)
#        my_dir = QtGui.QFileDialog.getExistingDirectory(self,"Open a folder",startfolder,QtGui.QFileDialog.ShowDirsOnly)
#        self.pathseismogram.setText(my_dir)
        
    @pyqtSlot()   
    def __test1__(self):            
        import image
        from mpl_toolkits.basemap import Basemap
        import pandas as pd
        filePath = QtGui.QFileDialog.getOpenFileName(self, 'Open file', '/Users/robertocabieces/Documents/obs_array')
        print(filePath)
        df=pd.read_csv(filePath,sep='\t')
        n=df.Name.count()
        Lat=[]
        Lon=[]
        for i in range(n):

            Lat.append(df.loc[i].Lat)
            Lon.append(df.loc[i].Lon)
        
        
        lf = self.LF.value()
        hf = self.HF.value()
        slim = self.Smax.value()  
          
         
        ARF=arf(filePath,lf,hf,slim)
        sstep = slim / len(ARF) 
        x = np.linspace(-1*slim, slim, (slim-(-1*slim)/sstep))
        y = np.linspace(-1*slim, slim, (slim-(-1*slim)/sstep))
        
        X, Y = np.meshgrid(x,y)
        
        self.grafico3.canvas.ax1.clear()           
        self.grafico3.canvas.ax1.contourf(X,Y,ARF,100,cmap=plt.cm.jet)
        ColorbarBase(self.grafico3.canvas.cax, cmap=plt.cm.jet,norm=Normalize(vmin=0, vmax=1))
        self.grafico3.canvas.draw()
        
        m = Basemap(projection='merc',llcrnrlat=min(Lat)-6,urcrnrlat=max(Lat)+6,\
            llcrnrlon=min(Lon)-6,urcrnrlon=max(Lon)+6,lat_ts=30,ax=self.grafico4.canvas.ax1)
        
        x,y=m(Lon,Lat)
        m.scatter(x, y, 75, color="r", marker="o", edgecolor="k", zorder=3)
        
        m.drawparallels(np.arange(min(Lat)-6,max(Lat)+6,4.),labels=[1,0,0,0],fontsize=10)
        m.drawmeridians(np.arange(min(Lon)-6,max(Lon)+6,4.),labels=[0,0,0,1],fontsize=10)
        #m.bluemarble()
        m.shadedrelief()
        #ColorbarBase(self.grafico4.canvas.cax, cmap=plt.cm.terrain,norm=Normalize(vmin=-6000, vmax=3000))
        n = Basemap(projection='merc',llcrnrlat=min(Lat)-0.5,urcrnrlat=max(Lat)+0.5,llcrnrlon=min(Lon)-0.5,urcrnrlon=max(Lon)+0.5,lat_ts=30,ax=self.grafico4.canvas.ax2)
        x,y=n(Lon,Lat)
        
        n.scatter(x, y, 75, color="r", marker="o", edgecolor="k", zorder=3)
        n.drawparallels(np.arange(min(Lat)-0.5,max(Lat)+0.5,0.5),labels=[1,0,0,0],fontsize=10)
        n.drawmeridians(np.arange(min(Lon)-0.5,max(Lon)+0.5,0.5),labels=[0,0,0,1],fontsize=10)
        #n.bluemarble()
        n.shadedrelief()
        
        #self.grafico4.canvas.fig.frameon=False
        self.grafico4.canvas.draw()

    @pyqtSlot()
    def __test2__(self):            
            #self.Labelcheck1.setText('Yes')    
            lf = self.LF1.value()
            hf = self.HF1.value()
            nc1 = self.NC1.value()
            nc2 = self.NC2.value()
            wd = self.WD.value()
            Path1=self.line1.text()
            Path2=self.line2.text()           
            wavelet(Path1,Path2,lf,hf,nc1,nc2,wd)
            
            seismogramplot=QtGui.QPixmap(masterpath+'/Wavelet_Analysis.png')
            waveletplotplot=QtGui.QPixmap(masterpath+'/CF_Analysis.png')
            scalogramplot=QtGui.QPixmap(masterpath+'/CWTOBS.png')

            scalogramplot = scalogramplot.scaled(800,900,transformMode=QtCore.Qt.SmoothTransformation)
            #QtCore.Qt.KeepAspectRatio
            self.plotSeismogram.setPixmap(seismogramplot)
            self.plotWavelet.setPixmap(waveletplotplot)
            self.plotScalogram.setPixmap(scalogramplot)
    @pyqtSlot()
    def __test3__(self):
            import matplotlib.dates as mdates
            xlocator = mdates.AutoDateLocator()
            Path_FK=self.path_FK.text()
            path_coords=self.path_coords.text() 
            
            lf2 = self.LF2.value()
            hf2 = self.HF2.value()
            MaxSlow = self.MaxSlow.value()
            SlowRes=self.SlowRes.value()
            TW=self.TW.value()
            Wfrac=self.Wfrac.value()
            stime=self.DateFK.text()
            DT=float(self.secFK.text())


            relpower,AZ,Slowness,T=FK(Path_FK,path_coords,stime,DT,lf2,hf2,MaxSlow,SlowRes,TW,Wfrac)
            TMIN=min(T)
            TMAX=max(T)  
            self.grafico5.canvas.ax1.clear()
            self.grafico5.canvas.ax1.scatter(T,relpower,s=10,c=relpower, alpha=0.5, marker=".",cmap=plt.cm.jet)
          
            self.grafico5.canvas.ax1.set_xlim(xmin=TMIN, xmax=TMAX)
            self.grafico5.canvas.ax1.set_ylabel("Rel.Power")
            self.grafico5.canvas.ax1.set_title("FK ANALYSIS")
            self.grafico5.canvas.ax1.xaxis.set_major_locator(xlocator)
            self.grafico5.canvas.ax1.xaxis.set_major_formatter(mdates.AutoDateFormatter(xlocator))
            
            
            self.grafico5.canvas.ax2.clear()
            self.grafico5.canvas.ax2.scatter(T,AZ,s=10,c=relpower, alpha=0.5, marker=".",cmap=plt.cm.jet)
            self.grafico5.canvas.ax2.set_ylabel("Azimuth [o]")
            self.grafico5.canvas.ax2.set_xlim(xmin=TMIN, xmax=TMAX)
            self.grafico5.canvas.ax2.xaxis.set_major_locator(xlocator)
            self.grafico5.canvas.ax2.xaxis.set_major_formatter(mdates.AutoDateFormatter(xlocator))
            
            
            self.grafico5.canvas.ax3.clear()
            self.grafico5.canvas.ax3.scatter(T,Slowness,s=10,c=relpower, alpha=0.5, marker=".",cmap=plt.cm.jet)
            self.grafico5.canvas.ax3.set_ylabel("Slowness [s/km]")
            self.grafico5.canvas.ax3.set_xlabel("Time [s]")
            self.grafico5.canvas.ax3.set_xlim(xmin=TMIN, xmax=TMAX)
            self.grafico5.canvas.ax3.xaxis.set_major_locator(xlocator)
            self.grafico5.canvas.ax3.xaxis.set_major_formatter(mdates.AutoDateFormatter(xlocator))
            #self.grafico5.canvas.ax3.autofmt_xdate()
            ColorbarBase(self.grafico5.canvas.cax, cmap=plt.cm.jet,norm=Normalize(vmin=0, vmax=1))
            
            
            def onclick(event):
                       
                if event.dblclick:
                    
                    if event.button == 1:
                                               
                        x1, y1 = event.xdata, event.ydata
                                    
                        
                        if self.FKbuttom.isChecked():
                            method = self.FKbuttom.text()
                            FKCoherence(Path_FK,path_coords,x1,lf2,hf2,MaxSlow,TW,0.005,method)
                            
                        if self.MTPbuttom.isChecked():
                            method = self.MTPbuttom.text()
                            FKCoherence(Path_FK,path_coords,x1,lf2,hf2,MaxSlow,TW,0.005,method)
                        
                    if event.button == 3:
                        self.grafico5.canvas.mpl_disconnect(cid)
                        print("Picking closed")
            
            cid=self.grafico5.canvas.mpl_connect('button_press_event', onclick)
            
            
            self.grafico5.canvas.draw()
            
            