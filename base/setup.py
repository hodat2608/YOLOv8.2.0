import sys,os
from pathlib import Path
current_dir = Path(__file__).resolve().parent.parent
ultralytics_main_dir = current_dir
sys.path.append(str(ultralytics_main_dir))
from ultralytics import YOLO
from base.constants import *
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import torch
import argparse
torch.cuda.empty_cache()
def main():
    parser = argparse.ArgumentParser(description="Train a YOLO model.")
    parser.add_argument('--config', type=str, default="training.yaml", help="Path to the YOLO configuration file.")
    parser.add_argument('--data', type=str, default="data.yaml", help="Path to the dataset configuration file.")
    parser.add_argument('--epochs', type=int, default=300, help="Number of training epochs.")
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--imgsz', type=int, default=640, help="Image size for training.")
    parser.add_argument('--batch', type=int, default=2, help="Batch size for training.")
    parser.add_argument('--cache', type=str2cache, nargs='?', default=False, help="Caching options: True (cache in RAM), 'disk' (cache on disk), or False (disable caching).")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], default=None, help="Device to use for training (cpu or cuda).")
    parser.add_argument('--project', type=str, default=None, help="Device to use for training (cpu or cuda).")
    parser.add_argument('--workers', type=int, default=2, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--name', type=str, default=None, help="Name of the training run. Used for creating a subdirectory within the project folder, where training logs and outputs are stored..")
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW','auto'], default='auto', help='optimizer')
    args = parser.parse_args()
    model = YOLO(args.config)
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train(data=args.data, epochs=args.epochs, patience=args.patience ,imgsz=args.imgsz, batch=args.batch, cache=args.cache, device=device,project=args.project,workers=args.workers,name=args.name,optimizer=args.optimizer)
    model.val()

if __name__ == '__main__':
    main()