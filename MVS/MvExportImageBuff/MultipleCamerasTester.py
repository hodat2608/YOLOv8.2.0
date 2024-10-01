# -- coding: utf-8 --
import sys
from tkinter import * 
from tkinter.messagebox import *
import _tkinter
import tkinter.messagebox
import tkinter as tk
import sys, os
from tkinter import ttk
sys.path.append("../MvImport")
from MvCameraControl_class import *
from CamOperation_classTester import *
from PIL import Image,ImageTk


def To_hex_str(num):
    chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    hexStr = ""
    if num < 0:
        num = num + 2**32
    while num >= 16:
        digit = num % 16
        hexStr = chaDic.get(digit, str(digit)) + hexStr
        num //= 16
    hexStr = chaDic.get(num, str(num)) + hexStr   
    return hexStr

class SetupTools():
    def __init__(self):
        self.deviceList = MV_CC_DEVICE_INFO_LIST()
        self.tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        self.obj_cam_operation = 0
        self.b_is_run = False
        self.nOpenDevSuccess = 0
        self.devList = []
        self.obj_cam_operation = []
        self.camObj = MvCamera()

    #ch:枚举相机 | en:enum devices
    def enum_devices(self):
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            tkinter.messagebox.showerror('show error','enum devices fail! ret = '+ To_hex_str(ret))

        #显示相机个数
        if deviceList.nDeviceNum == 0:
            tkinter.messagebox.showinfo('show info','find no device!')

        print ("Find %d devices!" % deviceList.nDeviceNum)

        for i in range(0, deviceList.nDeviceNum):
            mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                print ("\ngige device: [%d]" % i)
                strModeName = ""
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                    strModeName = strModeName + chr(per)
                print ("device model name: %s" % strModeName)

                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                print ("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
                self.devList.append("Gige["+str(i)+"]:"+str(nip1)+"."+str(nip2)+"."+str(nip3)+"."+str(nip4))
            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                print ("\nu3v device: [%d]" % i)
                strModeName = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                    if per == 0:
                        break
                    strModeName = strModeName + chr(per)
                print ("device model name: %s" % strModeName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber = strSerialNumber + chr(per)
                print ("user serial number: %s" % strSerialNumber)
                self.devList.append("USB["+str(i)+"]"+str(strSerialNumber))
    
        #ch:打开相机 | en:open device
    def open_device(self,i):
        if True == self.b_is_run:
            tkinter.messagebox.showinfo('show info','Camera is Running!')
            return
        self.obj_cam_operation.append(CameraOperation(self.camObj,self.deviceList,i))
        ret = self.obj_cam_operation[self.nOpenDevSuccess].Open_device()
        if  0!= ret:
            self.obj_cam_operation.pop()
        else:
            print(str(self.devList[i]))
            self.nOpenDevSuccess = self.nOpenDevSuccess + 1
        if 4 == self.nOpenDevSuccess:
            self.b_is_run = True    
        print("nOpenDevSuccess = ",self.nOpenDevSuccess) 

    # ch:开始取流 | en:Start grab image
    def start_grabbing(self,i):
        lock=threading.Lock() #申请一把锁
        ret = 0
        ret = self.obj_cam_operation[i].Start_grabbing(i,lock)
        if 0 != ret:
            tkinter.messagebox.showerror('show error','camera:'+ str(i) +',start grabbing fail! ret = '+ To_hex_str(ret))

    # ch:停止取流 | en:Stop grab image
    def stop_grabbing(self,i):
        ret = self.obj_cam_operation[i].Stop_grabbing()
        if 0 != ret:    
            tkinter.messagebox.showerror('show error','camera:'+ str(i) +'stop grabbing fail!ret = '+ To_hex_str(ret))

    # ch:关闭设备 | Close device   
    def close_device(self,i):
        ret = self.obj_cam_operation[i].Close_device()
        if 0 != ret:
            tkinter.messagebox.showerror('show error','camera:'+ str(i) + 'close deivce fail!ret = '+ To_hex_str(ret))
            self.b_is_run = True 
            return
        self.b_is_run = False 


a = SetupTools() 
a.enum_devices() 
a.open_device(0)