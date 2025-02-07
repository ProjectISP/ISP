#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 08:05:58 2025

@author: andres
"""

from PyQt5 import QtWidgets

from isp.Gui.Frames.uis_frames import UiPPSDsSaveFigure

class SaveFigureDialog(QtWidgets.QDialog, UiPPSDsSaveFigure):
    def __init__(self, figure):
        super(SaveFigureDialog, self).__init__()
        self.setupUi(self)
        
        self.figure = figure
        
        self.pushButton_2.clicked.connect(self.save_figure)
    
    def save_figure(self):
        output_path = QtWidgets.QFileDialog.getSaveFileName()[0]
        
        # Apply settings
        self.figure.set_figheight(self.doubleSpinBox.value())
        self.figure.set_figwidth(self.doubleSpinBox_2.value())

        self.figure.suptitle(self.lineEdit.text())
        self.figure.axes[-1].set_xlabel(self.lineEdit_2.text())
        self.figure.axes[-1].set_ylabel(self.lineEdit_3.text())
        
        self.figure.subplots_adjust(left=self.doubleSpinBox_3.value(), bottom=self.doubleSpinBox_6.value(),
                                    right=self.doubleSpinBox_4.value(), top=self.doubleSpinBox_5.value())
        
        format_ = self.comboBox.currentText()         
        self.figure.savefig(output_path + format_, dpi=self.spinBox.value(), format=format_[1:])