from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer, QCoreApplication
from PyQt5.QtWidgets import QApplication, QWidget, QFrame, QVBoxLayout
import sys
import time
import numpy as np
import socket

class LCDControl(QWidget):
    def __init__(self,app, m, maskimg):
        super(LCDControl,self).__init__()
        self.app = app
        self.m = m
        self.y = np.zeros((m,1))

        #window.show()

        self.label = QtWidgets.QLabel(self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.setContentsMargins(0,0,0,0)


        desktop = QtWidgets.QApplication.desktop()
        #controlled_lcd = desktop.screenGeometry(1)
        self.setGeometry(desktop.availableGeometry(1))
        self.showFullScreen()

        #img = QtGui.QPixmap('./img/sample_1024pixel.bmp').scaled(self.label.width(),self.label.height())
        #self.label.setPixmap(img)
        #label.resize(500,500)
        #label.show()
        self.img = maskimg

        print(self.img[0,0,:])
        self.count = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.setimg)
        self.show()

        img_np = (self.img[:,:,0] > 0.5).astype(np.uint8)*255
        img_qt = QtGui.QImage(img_np.copy(), img_np.shape[0], img_np.shape[1], QtGui.QImage.Format_Grayscale8)
        img_qpixmap = QtGui.QPixmap.fromImage(img_qt)
        img_qpixmap = img_qpixmap.scaled(self.label.width(), self.label.height())
        self.label.setPixmap(img_qpixmap)

        self.socket_tcp = initSensorCommunication()


        self.timer.start(100)
        app.exit(app.exec())
        #app.closeAllWindows()

    def init
    def setimg(self):
        #print(self.count)
        
        img_np = (self.img[:,:,self.count] > 0.5).astype(np.uint8)*255
        print(self.count, " ", img_np[0,0])
        img_qt = QtGui.QImage(img_np.copy(), img_np.shape[0], img_np.shape[1], QtGui.QImage.Format_Grayscale8)
        img_qpixmap = QtGui.QPixmap.fromImage(img_qt)
        img_qpixmap = img_qpixmap.scaled(self.label.width(), self.label.height())
        self.label.setPixmap(img_qpixmap)
        if self.count == 0:
            pass
        else:
            self.y[self.count-1,0] = GetSensorData(self.socket_tcp)
        self.count += 1
        
        if self.count == self.m:
            self.y[self.m-1,0] = GetSensorData(self.socket_tcp)
            #print(self.y)
            self.timer.stop()
            self.app.closeAllWindows()

    def gety(self):
        return self.y


    def resety(self):
        self.y = None
            


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
    socket_tcp.sendall(b'start')
    #print("send start signal to sensor")
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
    app = QApplication(sys.argv)
    maskimg = np.ones((30,40,30))
    #maskimg[:,20:40,3:5] = np.zeros((30,20,2))
    
    maskimg[:,:,0] = np.zeros((30,40))
    maskimg[:,:,4] = np.zeros((30,40))
    maskimg[:,:,9] = np.zeros((30,40))
    maskimg[:,:,7] = np.zeros((30,40))
    maskimg[:,:,12] = np.zeros((30,40))
    maskimg[:,:,13] = np.zeros((30,40))
    maskimg[:,:,18] = np.zeros((30,40))
    maskimg[:,:,24] = np.zeros((30,40))
    a = LCDControl(app,30,maskimg)
    y = a.gety()
    for i in range(maskimg.shape[2]):
        print(i," ",y[i,0])
    print("finished")

