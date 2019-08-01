# -*- coding: utf-8 -*-

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QFrame
from PyQt5.QtCore import Qt
import numpy as np
import sys
import serial
import serial.tools.list_ports
import cv2
import socket 
import time
import sys


class LcdDisplayWindow(QWidget):
    def __init__(self):
        super(LcdDisplayWindow, self).__init__()
        self.initUI()

    def initUI(self):
        # self.setWindowFlags(Qt.FramelessWindowHint)
        self.lb = QLabel("test")
        self.lb.setFrameStyle(QFrame.NoFrame)

        self.button = QPushButton("push")

        self.layout = QVBoxLayout()
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.lb)
        # self.layout.addWidget(self.button)
        self.setLayout(self.layout)

        img_np = (np.random.randn(30, 40) > 0.5).astype(float)
        print(img_np)
        # img_np = np.random.randn(30,40).astype(float)*255
        print(img_np.shape)

        # self.show()

        self.desktop = QtWidgets.QApplication.desktop()
        self.screen_count = self.desktop.screenCount()
        print(self.screen_count)

        self.controlled_lcd = self.desktop.screenGeometry(1)
        self.primary_screen = self.desktop.screenGeometry(0)
        print(self.controlled_lcd.width(), self.controlled_lcd.height())
        self.lcd_rect = self.desktop.availableGeometry(1)
        self.setGeometry(self.controlled_lcd)

        self.showFullScreen()
        self.setimage(img_np)
        # sys.exit(self.exec_())

    def setimage(self, img_np):
        img_qt = QtGui.QImage(
            img_np.copy(), img_np.shape[0], img_np.shape[1], QtGui.QImage.Format_Grayscale8)
        img_qpixmap = QtGui.QPixmap.fromImage(img_qt)
        img_qpixmap = img_qpixmap.scaled(self.lb.width(), self.lb.height())
        print(self.lb.width(), self.lb.height())
        self.lb.setPixmap(img_qpixmap)


def initSensorCommunication():
    SERVER_IP = "172.20.122.220" #树莓派的IP地址
    SERVER_PORT = 8888
    
    print("Starting socket: TCP...")
    server_addr = (SERVER_IP, SERVER_PORT)
    socket_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    while True:
        try:
            print("Connecting to server @ %s:%d..." %(SERVER_IP, SERVER_PORT))
            socket_tcp.connect(server_addr)
            break
        except Exception:
            print("Can't connect to server,try it latter!")
            time.sleep(1)
            continue
    print("Sensor onnected!")
    return socket_tcp

def GetSensorData(socket_tcp):
    ''' serial
    portx="/dev/cu.usbserial-1450"
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
    '''

    # socket
    data = None
    while data == None:
        #try:
        data = socket_tcp.recv(512)
        data = data.split(b'.')
        
        if len(data)==0:
            data = None
            continue
        else:
            data = int(data[0])
        '''except Exception as e:
            print(e)
            socket_tcp.close()
            socket_tcp=None
            sys.exit(1)'''

    return data




if __name__ == "__main__":
    #app = QtWidgets.QApplication(sys.argv)
    #a = LcdDisplayWindow()
    #img1 = np.random.randn(30,40)
    #img = (img1 > 0.5).astype(float)
    #print(img)
    # a.setimage(img)
    #cv2.imshow("img1",img1)
    #cv2.imshow("img", img)
    #cv2.waitKey(0)
    # while True:
    #    print(GetSensorData())
    socket_tcp = initSensorCommunication()
    while True:
        print(GetSensorData(socket_tcp))
