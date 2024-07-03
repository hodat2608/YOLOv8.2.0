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
model = YOLO("C:/Users/CCSX009/Documents/ultralytics-main/best.pt")
print('load model successfully')
def detect_images(model):
    # source = "C:/Users/CCSX009/Documents/ultralytics-main/2024-03-05_00-01-31-398585-C1.jpg"
    image_paths = glob.glob(f"C:/Users/CCSX009/Documents/yolov5/test_image/camera1/*.jpg")
    if len(image_paths) == 0:
        pass
    else:
        for filename in image_paths:
            t1 = time.time()
            results = model(filename,imgsz=608,conf=0.2)
            list_remove = []
            # print(results)
            for result in results:
                print(result.boxes)
                bos = result.boxes.cpu().numpy()
                xywh_list = bos.xywh.tolist()
                cls_list = bos.cls.tolist()
                conf_list = bos.conf.tolist()
                names_dict = result.names
                print(names_dict)
                for xywh, cls, conf in zip(xywh_list, cls_list, conf_list):
                    class_name = names_dict[int(cls)]  
                    print(f'{xywh[2]}--{xywh[3]}--{cls}--{conf}--{class_name}')
                    print('-------------------------------------')              
                pcs = np.squeeze(result.extract_np(list_remove=list_remove))
                t2 = time.time() - t1
                time_processing = str(int(t2*1000)) + 'ms'
                if pcs.dtype != np.uint8:
                    pcs = pcs.astype(np.uint8)
                if pcs.shape[2] == 1:
                    pcs = np.concatenate([pcs, pcs, pcs], axis=2)
                img = Image.fromarray(pcs.astype('uint8'), 'RGB')
                img.show()
                os.remove(filename)
                print(time_processing)
                result.show()
    
while True:
    detect_images(model)
