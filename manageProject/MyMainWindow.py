from PyQt4 import *
from PyQt4 import Qt, QtGui, uic, QtCore
from PyQt4.QtCore import pyqtSlot, pyqtSignal, QThread, QEventLoop, QObject
from arklink import *
import subprocess as S
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import os
ventana_principal = "MainWindow.ui"  

MainWindow, _ = uic.loadUiType(ventana_principal)


class MyMainWindow(QtGui.QMainWindow, MainWindow):

    def __init__(self, parent):
        QtGui.QMainWindow.__init__(self)
        MainWindow.__init__(self)
        self.setupUi(self)
        # App
        self.app = parent

        self.retrieveSeisComp3.clicked.connect(self.retrieve)
        #self.pathseismogram.setText("/Users/robertocabieces/Desktop/GUIPYTHON/ArrayProcessing/260/RAW")
        self.client.addItems(["Server","BGR","EMSC","ETH","GEONET","GFZ","ICGC","INGV","IPGP","IRIS","ISC","KNMI","KOERI","LMU","NCEDC","NIEP","NOA","ODC","ORFEUS","RESIF","SECEDC","TEXNET","USGS","USP"])
        self.restrievewaveforms.clicked.connect(self.retrievewaveforms)
        self.OF2.clicked.connect(self.load)
        #self.infostations.clicked.connect(self.infostationsload)
        #self.infoevents.clicked.connect(self.infoeventsload)
    
    
    
    def retrieve(self):
    ##Seiscomp3 retrieve data from a database
        timeini=self.dateTimeEvent1.dateTime().toString("yyyy-MM-dd hh:mm:ss")
        timefin=self.dateTimeEvent2.dateTime().toString("yyyy-MM-dd hh:mm:ss")
                    
        #Idevent            
        EventId=self.EventId.text()
        SC3DataBase=self.SC3DataBase.text()
        SDS=self.SDS.text()
        OutputFolder=self.output.text()
        #command="cd "+path+";"+"python main.py"
        #caso1
        if self.Option1.isChecked():
            #command1="SC3p1 "+" "+SC3DataBase+" "+SDS
            ##Step1 go to thedowload data
            ##Step2 run the command
            ##Step3 move the data
            command1="echo"+" "+"SC3p1 "+" "+SC3DataBase+" "+SDS
            S.Popen(command1,shell=True)
        #caso2 
        if self.Option2.isChecked():
            command1="SC3p2 "+"SC3DataBase "+"SDS "+"timeini "+"timefin"
             
            S.Popen(command1,shell=True)
            
    def load(self):
        startfolder=os.getcwd
        startfolder=str(startfolder)
        my_dir = QtGui.QFileDialog.getExistingDirectory(self,"Open a folder",startfolder,QtGui.QFileDialog.ShowDirsOnly)
        self.output2.setText(my_dir)
                    
    def retrievewaveforms(self):
        

        output=self.output2.text()
        dateTimeEventclient1=self.dateTimeEventclient1.dateTime().toString("yyyy-MM-dd hh:mm:ss")
        dateTimeEventclient2=self.dateTimeEventclient2.dateTime().toString("yyyy-MM-dd hh:mm:ss")
        time1=dateTimeEventclient1[0:9]+"T"+dateTimeEventclient1[11:19]
        time2=dateTimeEventclient2[0:9]+"T"+dateTimeEventclient2[11:19]
        server=self.client.currentText()
        net=self.Net2.text()
        sta=self.Sta2.text()
        loc=self.Loc2.text()
        channel=self.Channel2.text()
        arklink(server,net,sta,loc,channel,time1,time2,output,write=True)
#        
    
        
        
        
        
        
        
        