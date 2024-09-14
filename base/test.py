import sys,os,glob,shutil
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
torch.cuda.empty_cache()

# # # Create a new YOLOv8n-OBB model from scratch
# # model = YOLO(r"C:\Users\CCSX009\Documents\label\yolov8n_obb.yaml")

# # # Train the model on the DOTAv2 dataset
# # if __name__ == '__main__':
# #     results = model.train(data=r"C:\Users\CCSX009\Documents\label\data_cfg.yaml", epochs=100, imgsz=640,device='cpu')
# #     metrics = model.val()

import os
import glob
import shutil

txt_path = r"C:\Users\CCSX009\Desktop\obb\New folder\2024-06-05_04-32-26-465559-C1.txt"
folder_img_path = r'C:\Users\CCSX009\Desktop\obb'

# Duyệt qua tất cả các file ảnh .jpg trong thư mục
for img_path in glob.glob(os.path.join(folder_img_path, '*.jpg')):
    tenf = os.path.basename(img_path)  # Lấy tên file ảnh
    
    # Đường dẫn mới cho file txt sẽ được sao chép
    new_txt_path = os.path.join(folder_img_path, tenf[:-3] + 'txt')
    
    # Sao chép file txt với tên mới
    shutil.copy(txt_path, new_txt_path)

print("Hoàn thành sao chép file txt với tên tương ứng cho từng ảnh.")

# # from ultralytics.data.converter import convert_dota_to_yolo_obb


# # convert_dota_to_yolo_obb(r"C:\Users\CCSX009\Downloads\dota8\dota8")
# import numpy as np

# def obb_to_4_points_normalized(class_id, x_center, y_center, width, height, angle, img_width, img_height):
#     # Tính nửa chiều rộng và nửa chiều cao
#     half_width = width / 2
#     half_height = height / 2
    
#     # Chuyển góc từ độ sang radian
#     angle_rad = np.deg2rad(angle)
    
#     # Tạo ma trận quay dựa trên góc nghiêng
#     rotation_matrix = np.array([
#         [np.cos(angle_rad), -np.sin(angle_rad)],
#         [np.sin(angle_rad), np.cos(angle_rad)]
#     ])
    
#     # Tọa độ tương đối của 4 đỉnh (theo chiều kim đồng hồ) trước khi xoay
#     corners = np.array([
#         [-half_width, -half_height],  # Bottom-left
#         [half_width, -half_height],   # Bottom-right
#         [half_width, half_height],    # Top-right
#         [-half_width, half_height]    # Top-left
#     ])
    
#     # Xoay các góc dựa trên ma trận quay
#     rotated_corners = np.dot(corners, rotation_matrix)
    
#     # Dịch các góc về vị trí x_center, y_center
#     final_corners = rotated_corners + np.array([x_center, y_center])
    
#     # Chia x cho img_width và y cho img_height để chuẩn hóa các tọa độ
#     normalized_corners = final_corners / np.array([img_width, img_height])
    
#     # Trả về kết quả với class_id và 4 tọa độ đỉnh (x1, y1, x2, y2, x3, y3, x4, y4)
#     return [class_id] + normalized_corners.flatten().tolist()

# # Ví dụ sử dụng với kích thước ảnh w = 1200, h = 1200
# obb_label = [0, 758.007233, 282.513061, 534.987392, 174.524347, -15.110780]
# img_width = 1200
# img_height = 1200
# four_points_label_normalized = obb_to_4_points_normalized(*obb_label, img_width, img_height)

# print(four_points_label_normalized)


# import os
# import numpy as np

# def obb_to_4_points_normalized(class_id, x_center, y_center, width, height, angle, img_width, img_height):
#     # Tính nửa chiều rộng và nửa chiều cao
#     half_width = width / 2
#     half_height = height / 2
    
#     # Chuyển góc từ độ sang radian
#     angle_rad = np.deg2rad(angle)
    
#     # Tạo ma trận quay dựa trên góc nghiêng
#     rotation_matrix = np.array([
#         [np.cos(angle_rad), -np.sin(angle_rad)],
#         [np.sin(angle_rad), np.cos(angle_rad)]
#     ])
    
#     # Tọa độ tương đối của 4 đỉnh (theo chiều kim đồng hồ) trước khi xoay
#     corners = np.array([
#         [-half_width, -half_height],  # Bottom-left
#         [half_width, -half_height],   # Bottom-right
#         [half_width, half_height],    # Top-right
#         [-half_width, half_height]    # Top-left
#     ])
    
#     # Xoay các góc dựa trên ma trận quay
#     rotated_corners = np.dot(corners, rotation_matrix)
    
#     # Dịch các góc về vị trí x_center, y_center
#     final_corners = rotated_corners + np.array([x_center, y_center])
    
#     # Chia x cho img_width và y cho img_height để chuẩn hóa các tọa độ
#     normalized_corners = final_corners / np.array([img_width, img_height])
    
#     # Trả về kết quả với class_id và 4 tọa độ đỉnh (x1, y1, x2, y2, x3, y3, x4, y4)
#     return [class_id] + normalized_corners.flatten().tolist()

# def process_txt_files(input_folder, output_folder, img_width, img_height):
#     # Đảm bảo thư mục output tồn tại
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # Duyệt qua tất cả các file .txt trong thư mục input
#     for txt_file in os.listdir(input_folder):
#         if txt_file.endswith('.txt'):
#             input_path = os.path.join(input_folder, txt_file)
#             output_path = os.path.join(output_folder, txt_file)

#             # Đọc nội dung file txt
#             with open(input_path, 'r') as file:
#                 lines = file.readlines()

#             # Mở file output để ghi dữ liệu đã được convert
#             with open(output_path, 'w') as out_file:
#                 for line in lines:
#                     line = line.strip()
#                     # Bỏ qua các dòng chứa "YOLO_OBB"
#                     if "YOLO_OBB" in line:
#                         continue
                    
#                     # Chuyển đổi từng dòng dữ liệu (class_id, x_center, y_center, width, height, angle)
#                     params = list(map(float, line.split()))
#                     class_id, x_center, y_center, width, height, angle = params
#                     converted_label = obb_to_4_points_normalized(class_id, x_center, y_center, width, height, angle, img_width, img_height)
                    
#                     # Ghi kết quả chuyển đổi vào file mới
#                     out_file.write(" ".join(map(str, converted_label)) + '\n')

#     print(f"Đã chuyển đổi tất cả các file txt trong {input_folder} sang dạng [class_index, x1, y1, x2, y2, x3, y3, x4, y4] và lưu vào {output_folder}.")

# # Đường dẫn tới thư mục chứa các file txt
# input_folder = r'C:\Users\CCSX009\Desktop\obb'
# output_folder = r'C:\Users\CCSX009\Desktop\obb\a'

# # Kích thước ảnh (w, h)
# img_width = 1200
# img_height = 1200

# # Gọi hàm để xử lý các file txt
# process_txt_files(input_folder, output_folder, img_width, img_height)

