#from PyQt4 import Qt, QtGui, uic, QtCore, QFont
#import sys
from PyQt4 import *
from PyQt4 import Qt, QtGui, uic, QtCore
from PyQt4.QtCore import pyqtSlot, pyqtSignal, QThread, QEventLoop, QObject
import os
from recursos_1_rc import *
import subprocess as S
from pathlib import Path
masterpath=Path(__file__).resolve().parent
masterpath=str(masterpath)
ventana_principal=masterpath+"/MainWindow0.ui"
MainWindow0, _ = uic.loadUiType(ventana_principal)


class MyMainWindow(QtGui.QMainWindow, MainWindow0):

    def __init__(self, parent):
        QtGui.QMainWindow.__init__(self)
        MainWindow0.__init__(self)
        self.setupUi(self)
        # App
        self.app = parent
                
        #self.actionCreatenewProject.triggered.connect(self.create)
        #self.ManageProject.clicked.connect(self.prueba)
        
        icon1=QtGui.QPixmap(masterpath+'/ImagenesISP/'+'02.png')
        icon2=QtGui.QPixmap(masterpath+'/ImagenesISP/'+'03.png')
        icon3=QtGui.QPixmap(masterpath+'/ImagenesISP/'+'04.png')
        icon4=QtGui.QPixmap(masterpath+'/ImagenesISP/'+'05.png')
        icon5=QtGui.QPixmap(masterpath+'/ImagenesISP/'+'01.png')
        
        iconLogo = QtGui.QPixmap(masterpath+'/ImagenesISP/'+'LOGO.png')
	
        self.labelManage.setPixmap(icon1)
        self.labelseismogram.setPixmap(icon2)
        self.labelearthquake.setPixmap(icon3)
        self.labelMTI.setPixmap(icon4)
        self.labelarray.setPixmap(icon5)
               
        self.LOGO.setPixmap(iconLogo)
        
        self.SeismogramAnalysis.clicked.connect(self.runSeismogram)
        self.ArrayAnalysis.clicked.connect(self.array)
        

    def runSeismogram(self):        
        path=masterpath+"/seismogramInspector/"
        command="cd "+path+";"+"python main.py"
        S.Popen(command,shell=True)
      
    def array(self):        
        path=masterpath+"/arrayanalysis/"
        command="cd "+path+";"+"python main.py"
        S.Popen(command,shell=True)

