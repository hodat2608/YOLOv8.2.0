import os
import numpy as np

def xywhr2xyxyxyxy(class_id, x_center, y_center, width, height, angle, img_width, img_height):
    half_width = width / 2
    half_height = height / 2
    angle_rad = np.deg2rad(angle)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    corners = np.array([
        [-half_width, -half_height],  
        [half_width, -half_height], 
        [half_width, half_height],   
        [-half_width, half_height]
    ])
    rotated_corners = np.dot(corners, rotation_matrix)
    final_corners = rotated_corners + np.array([x_center, y_center])
    normalized_corners = final_corners / np.array([img_width, img_height])
    return [int(class_id)] + normalized_corners.flatten().tolist()

def process_txt_files(input_folder, output_folder, img_width, img_height):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for txt_file in os.listdir(input_folder):
        if txt_file.endswith('.txt'):
            input_path = os.path.join(input_folder, txt_file)
            output_path = os.path.join(output_folder, txt_file)
            with open(input_path, 'r') as file:
                lines = file.readlines()
            with open(output_path, 'w') as out_file:
                for line in lines:
                    line = line.strip()
                    if "YOLO_OBB" in line:
                        continue
                    params = list(map(float, line.split()))
                    class_id, x_center, y_center, width, height, angle = params
                    converted_label = xywhr2xyxyxyxy(class_id, x_center, y_center, width, height, angle, img_width, img_height)
                    out_file.write(" ".join(map(str, converted_label)) + '\n')
