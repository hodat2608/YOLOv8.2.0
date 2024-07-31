import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent.parent
ultralytics_main_dir = current_dir
sys.path.append(str(ultralytics_main_dir))
from ultralytics import YOLO
from PIL import Image
import cv2
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
from pathlib import Path
import cv2

def detect_video(source):
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(source)
    def update_frame():
        ret, frame = cap.read()
        if not ret:
            return
        # results = model.track(frame,show=True)
        # annotated_frame = results[0].plot()
        # boxes_dict = results[0].boxes.cpu().numpy()
        # xywh_list = boxes_dict.xywh.tolist()
        # cls_list = boxes_dict.cls.tolist()
        # conf_list = boxes_dict.conf.tolist()
        # print(f'{xywh_list}-----{cls_list}------{conf_list}')
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        root.after(2, update_frame)

    root = tk.Tk()
    root.title("YOLOv8 Video Detection")

    label = Label(root)
    label.pack()

    update_frame()
    root.mainloop()

if __name__ == "__main__":
    source = r"C:\Users\CCSX009\Videos\y2mate.com - TEQBALL  Rally of the Year_1080p.mp4"
    detect_video(source)
