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
import argparse
# model = YOLO("C:/Users/CCSX009/Documents/ultralytics-main/training.yaml")
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model.to(device)
# path = model.export(format="onnx")  # export the model to ONNX format
# if __name__ == '__main__':
#     model.train(data="data.yaml", epochs=100, imgsz=468, batch=32, device='cpu')
#     metrics = model.val()

def main():
    # Tạo đối tượng ArgumentParser
    parser = argparse.ArgumentParser(description="Train a YOLO model.")
    
    # Thêm các đối số mà bạn muốn truyền vào từ dòng lệnh
    parser.add_argument('--config', type=str, default="training.yaml",
                        help="Path to the YOLO configuration file.")
    parser.add_argument('--data', type=str, default="data.yaml",
                        help="Path to the dataset configuration file.")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('--imgsz', type=int, default=468, help="Image size for training.")
    parser.add_argument('--batch', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default=None, help="Device to use for training (cpu or cuda).")
    parser.add_argument('--project', type=str, default=None,help="Device to use for training (cpu or cuda).")
    
    # Parse các đối số từ dòng lệnh
    args = parser.parse_args()
    
    # Khởi tạo model YOLO với file config
    model = YOLO(args.config)
    
    # Xác định thiết bị (device) để sử dụng
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Bắt đầu quá trình huấn luyện
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=device,project=args.project)
    
    # Đánh giá mô hình sau khi huấn luyện
    metrics = model.val()
    print(metrics)

if __name__ == '__main__':
    main()