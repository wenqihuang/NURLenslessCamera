# -*- coding: utf-8 -*-

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QFrame
from PyQt5.QtCore import Qt
import numpy as np
import sys
import serial
import serial.tools.list_ports


class LcdDisplayWindow(QWidget):
    def __init__(self):
        super(LcdDisplayWindow,self).__init__()
        self.initUI()
        
    def initUI(self):
        #self.setWindowFlags(Qt.FramelessWindowHint)
        self.lb = QLabel("test")
        self.lb.setFrameStyle(QFrame.NoFrame)

        self.button = QPushButton("push")

        self.layout = QVBoxLayout()
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.addWidget(self.lb)
        #self.layout.addWidget(self.button)
        self.setLayout(self.layout)

        
        img_np = np.random.randn(20,80)
        print(img_np.shape)
        
        #self.show()

        self.desktop = QtWidgets.QApplication.desktop()
        self.screen_count = self.desktop.screenCount()
        print(self.screen_count)

        self.controlled_lcd = self.desktop.screenGeometry(1)
        self.primary_screen = self.desktop.screenGeometry(0)
        print(self.controlled_lcd.width(),self.controlled_lcd.height())
        self.lcd_rect = self.desktop.availableGeometry(1)
        self.setGeometry(self.controlled_lcd)

        self.showFullScreen()
        self.setimage(img_np)
        sys.exit(app.exec_())

    def setimage(self,img_np):
        img_qt = QtGui.QImage(img_np,img_np.shape[0],img_np.shape[1],QtGui.QImage.Format_Grayscale8)
        img_qpixmap = QtGui.QPixmap.fromImage(img_qt)
        img_qpixmap = img_qpixmap.scaled(self.lb.width(),self.lb.height())
        self.lb.setPixmap(img_qpixmap)


def GetSensorData():
    portx="/dev/cu.usbserial-1440"
    bps=9600
    timex=5
    ser=serial.Serial(portx,bps,timeout=timex)
    
    while True:
        try:
            data=ser.readline().decode('ascii')[:-6]
            data = int(data)
            break
        except ValueError:
            continue
        
    ser.close()
    return data




if __name__ == "__main__":
    #app = QtWidgets.QApplication(sys.argv)
    #a = LcdDisplayWindow()
    while True:
        print(GetSensorData())
