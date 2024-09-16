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

# model = YOLO(r"C:\Users\CCSX009\Desktop\obb\20240912\weights\best.pt")
# results = model(r"C:\Users\CCSX009\Downloads\a.jpg")

# for result in results:
#     obb = result.obb.cpu().numpy()
#     result.show()
#     print(obb)
import numpy as np
import cv2
from ultralytics import YOLO
from norfair import Detection, Tracker

# Load YOLOv8 model
model = YOLO(r"C:\Users\CCSX009\Documents\ultralytics-main\best_obb.pt")

# Initialize Norfair Tracker


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

def yolo_to_norfair(result):
    """
    Convert YOLOv8 result format to Norfair's Detection format.
    """
    detections = []
    for obb in result.obb:
        # Kiểm tra nếu obb có đủ giá trị
        if len(obb) >= 5:
            class_id, x_center, y_center, width, height, angle = obb[:6]
            points = np.array([[x_center, y_center]])  # Center point for Norfair tracking
            detection = Detection(points=points, scores=[1.0])  # Set a default score (or use actual confidence)
            detections.append(detection)
        else:
            print(f"Invalid obb format: {obb}")
    return detections
# Load and process the image
image_path = r"C:\Users\CCSX009\Downloads\a.jpg"
results = model(image_path)
tracker = Tracker(distance_function=euclidean_distance, distance_threshold=30)
# Run tracking
for result in results:
    obb = result.obb.cpu().numpy()
    
    # Convert YOLO result to Norfair detection format
    detections = yolo_to_norfair(result)
    
    # Update the tracker with detections
    tracked_objects = tracker.update(detections)
    
    # Visualize results
    for tracked_object in tracked_objects:
        tracking_id = tracked_object.id
        x_center, y_center = tracked_object.estimate[0]
        
        # Draw the bounding box and the tracking ID on the image
        cv2.putText(result.orig_img, f"ID: {tracking_id}", (int(x_center), int(y_center)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print(f"Object with ID {tracking_id} at position ({x_center}, {y_center})")
    
    # Show image with tracking
    result.show()

