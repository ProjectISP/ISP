import sys

from MyMainWindow0 import *
from PyQt4 import QtGui


app = QtGui.QApplication(sys.argv)


#Iniciamos número de ventanas del programa
window = MyMainWindow(app)
#window2 = MyDialog1(app)

#window.resultChanged.connect(window2.imprimeRes)

window.show()
#window2.show()
sys.exit(app.exec_())