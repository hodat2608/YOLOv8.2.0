import sys
from pathlib import Path
import os
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

model = YOLO(r"C:\Users\CCSX009\Desktop\obb\20240912\weights\best.pt")
results = model(r"C:\Users\CCSX009\Downloads\a.jpg")

for result in results:
    obb = result.obb.cpu().numpy()
    result.show()
    print(obb)

