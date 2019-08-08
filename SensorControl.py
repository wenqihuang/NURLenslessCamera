#!/usr/bin/env python
# encoding: utf-8
# THIS FILE IS FOR RESPBERRY PI TO CONTROL SENSOR AND SEND DATA TO MAC
import smbus
import time
import serial
import socket
import sys


#BH1750 ^|  ^}^`
__DEV_ADDR=0x23

# ^n  ^h   ^w
__CMD_PWR_OFF=0x00  # ^e  ^|
__CMD_PWR_ON=0x01   #  ^` ^|
__CMD_RESET=0x07    # ^g^m
__CMD_CHRES=0x10    # ^l^a     ^x ^h^f    ^n^g  ^`  ^k
__CMD_CHRES2=0x11   # ^l^a     ^x ^h^f    ^n^g     ^o2  ^`  ^k
__CMD_CLHRES=0x13   # ^l^a     ^n ^h^f    ^n^g  ^`  ^k
__CMD_THRES=0x20    #  ^`     ^x ^h^f    ^n^g
__CMD_THRES2=0x21   #  ^`     ^x ^h^f    ^n^g     ^o2
__CMD_TLRES=0x23    #  ^`    ^h^f    ^n^g
__CMD_SEN100H=0x42  # ^a  ^u^o   100%,  ^x  ^m
__CMD_SEN100L=0X65  # ^a  ^u^o   100%  ^l  ^n  ^m
__CMD_SEN50H=0x44   #50%
__CMD_SEN50L=0x6A   #50%
__CMD_SEN200H=0x41  #200%
__CMD_SEN200L=0x73  #200%
__CMD_SENMAXH=0x47
__CMD_SENMAXL=0x7E

bus=smbus.SMBus(1)
bus.write_byte(__DEV_ADDR,__CMD_PWR_ON)
bus.write_byte(__DEV_ADDR,__CMD_RESET)
bus.write_byte(__DEV_ADDR,__CMD_SENMAXH)
bus.write_byte(__DEV_ADDR,__CMD_SENMAXL)
bus.write_byte(__DEV_ADDR,__CMD_PWR_OFF)
def getIlluminance():
   # while True:
    bus.write_byte(__DEV_ADDR,__CMD_PWR_ON)
    bus.write_byte(__DEV_ADDR,__CMD_THRES2)
    time.sleep(0.44)
    res=bus.read_word_data(__DEV_ADDR,0)

   #read_word_data
    res=((res>>8)&0xff)|(res<<8)&0xff00
    #res=round(res/(2*1.2),2)
    result=" ^e^i ^e       : "+str(res)+"lx"
    #print(res)
    return res


HOST_IP = "172.20.122.220" #  ^q ^n^s    ^z^dIP ^|  ^}^`
HOST_PORT = 8888
print("Starting socket: TCP...")
#1.create socket object:socket=socket.socket(family,type)
socket_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket_tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
print("TCP server listen @ %s:%d!" %(HOST_IP, HOST_PORT) )
host_addr = (HOST_IP, HOST_PORT)
#2.bind socket to addr:socket.bind(address)
socket_tcp.bind(host_addr)
#3.listen connection request:socket.listen(backlog)
socket_tcp.listen(1)
#4.waite for client:connection,address=socket.accept()
socket_con, (client_ip, client_port) = socket_tcp.accept()
print("Connection accepted from %s." %client_ip)
#socket_con.send("Welcome to RPi TCP server!")


#portx='/dev/serial1'
#bps=9600
#ser=serial.Serial(portx,bps)
while True:
    data = getIlluminance()
    #ser.write(data)
    socket_con.send(str(data)+'.')
    #time.sleep(0.2)