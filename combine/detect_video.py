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
import argparse
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors

def detect_video(source):
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(source)
    def update_frame():
        ret, frame = cap.read()
        if not ret:
            return
        results = model.track(frame,show=True)
        annotated_frame = results[0].plot()
        boxes_dict = results[0].boxes.cpu().numpy()
        xywh_list = boxes_dict.xywh.tolist()
        cls_list = boxes_dict.cls.tolist()
        conf_list = boxes_dict.conf.tolist()
        print(f'{xywh_list}-----{cls_list}------{conf_list}')
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        root.after(1, update_frame)

    root = tk.Tk()
    root.title("YOLOv8 Video Detection")

    label = Label(root)
    label.pack()

    update_frame()
    root.mainloop()

if __name__ == "__main__":
    source = r"C:\Users\CCSX009\Videos\y2mate.com - TEQBALL  Rally of the Year_1080p.mp4"
    detect_video(source)

    '''
                    A(1252,787)       B(2298,803)
                      ----------------                  
                     /              /         
                    /              / 
                   /              /        Origin Frame Size (pixel) 
                  /              /
                 /              /
                ----------------
              D(-550,2159)      C(5039,2159)

              
                    A'(0,0)           B'(24,0)
                      ----------------
                     /              /         
                    /              / 
                   /              /        Origin Actual Size (meter)
                  /              /
                 /              /
                ----------------
              D'(0,249)        C'(24,249)
'''
# if __name__ == "__main__":
#     source = r"C:\Users\CCSX009\Videos\vecteezy_car-and-truck-traffic-on-the-highway-in-europe-poland_7957364.mp4"
#     target_path = r"C:\Users\CCSX009\Videos"
#     run(source_path=source, video_path=source, target_path=target_path)