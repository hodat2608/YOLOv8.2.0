import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent.parent
ultralytics_main_dir = current_dir
sys.path.append(str(ultralytics_main_dir))
from ultralytics import YOLO
import ultralytics
from PIL import Image
import numpy as np
import cv2
import glob,os,time
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import torch
print(ultralytics.__file__)
model = YOLO(r"C:\Users\CCSX009\Documents\ultralytics-main\best.pt")

# torch.cuda.set_device(0)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model1 = YOLO('/your/model/path1/', task='detect')
# model2 = YOLO('/your/model/path2/', task='detect')
# model.to(device=device)
# print('load model successfully')

def detect_images(model):
    image_paths = glob.glob(f"C:/Users/CCSX009/Documents/yolov5/test_image/camera1/*.jpg")
    try:
        if len(image_paths) == 0:
            pass
        else:
            for filename in image_paths:
                t1 = time.time()
                results = model(filename,imgsz=608,conf=0.01)
                results.show()
                # list_remove = [8,10,11,12,13,14,15,16]
                # dictionary=[]
                # for result in results:
                #     print(result.orig_shape)
                #     bos = result.boxes.cpu().numpy()
                #     xywh_list = bos.xywh.tolist()
                #     cls_list = bos.cls.tolist()
                #     conf_list = bos.conf.tolist()
                #     names_dict = result.names
                
                #     for xywh, cls, conf in zip(xywh_list, cls_list, conf_list):
                #         class_name = names_dict[int(cls)]  
                #         if int(cls) in list_remove: 
                #             continue
                #         dictionary.append((xywh[0], xywh[1], xywh[2], xywh[3], int(cls)))
                #     pcs1 = np.squeeze(result.extract_npy(list_remove=list_remove))
                #     pcs = np.squeeze(result.render_x_y(dictionary=dictionary))
                #     output_image = cv2.cvtColor(pcs, cv2.COLOR_BGR2RGB)
                #     t2 = time.time() - t1
                #     time_processing = str(int(t2*1000)) + 'ms'
                #     if pcs.dtype != np.uint8:
                #         pcs = pcs.astype(np.uint8)
                #     if pcs.shape[2] == 1:
                #         pcs = np.concatenate([pcs, pcs, pcs], axis=2)
                #     img = Image.fromarray(output_image.astype('uint8'), 'RGB')
                #     img.show()
                #     os.remove(filename)
                #     print(time_processing)
    except Exception as e :
        print(e)

while True:
    detect_images(model)
