import sys,os
from pathlib import Path
current_dir = Path(__file__).resolve().parent.parent
ultralytics_main_dir = current_dir
sys.path.append(str(ultralytics_main_dir))
from ultralytics import YOLO
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import torch
# Load a model
model = YOLO("C:/Users/CCSX009/Documents/ultralytics-main/training.yaml")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
# path = model.export(format="onnx")  # export the model to ONNX format

if __name__ == '__main__':
    model.train(data="data.yaml", epochs=300, imgsz=608, batch=10, device='cpu')
    metrics = model.val()
