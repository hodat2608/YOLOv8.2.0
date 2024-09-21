import os
import numpy as np
import cv2
import shutil
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


def xyxyxyxy_to_xywhr_low_tolerance(class_id,x1, y1, x2, y2, x3, y3, x4, y4, img_width, img_height):
        '''
        dung sai góc alpha giữa 2 lần convert thấp, tọa độ không đổi, tuy nhiên kích thước width & height sẽ có sự chênh lệch nhỏ
        '''
        x_center_norm = (x1 + x2 + x3 + x4) / 4
        y_center_norm = (y1 + y2 + y3 + y4) / 4
        width_norm = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        height_norm = np.sqrt((x4 - x1)**2 + (y4 - y1)**2)
        angle_rad = np.arctan2(y2 - y1, x2 - x1)
        angle_deg = -(np.degrees(angle_rad))
        x_center = x_center_norm * img_width
        y_center = y_center_norm * img_height
        width = width_norm * img_height
        height = height_norm * img_width
        return class_id, x_center, y_center, width, height, angle_deg

def xyxyxyxy_to_xywhr_high_tolerance(class_id, x1, y1, x2, y2, x3, y3, x4, y4, img_width, img_height):
    '''
    dung sai góc alpha giữa 2 lần convert cao, tọa độ không đổi, tuy nhiên kích thước width & height không có sự chênh lệch
    '''
    points = np.array([
        [x1 * img_width, y1 * img_height],
        [x2 * img_width, y2 * img_height],
        [x3 * img_width, y3 * img_height],
        [x4 * img_width, y4 * img_height]
    ])
    center = points.mean(axis=0)
    width = np.linalg.norm(points[1] - points[0])
    height = np.linalg.norm(points[3] - points[0]) 
    angle_deg = (np.degrees(np.arctan2(points[1][1] - points[0][1], points[1][0] - points[0][0])) % 180)
    return class_id, center[0],center[1], width, height, angle_deg

def format_params(des_path,progress_label):
    input_folder = des_path
    os.makedirs(os.path.join(input_folder,'instance'),exist_ok=True)
    output_folder = (os.path.join(input_folder,'instance'))
    total_fl = len(des_path) 
    for index,txt_file in enumerate(os.listdir(input_folder)):
        if txt_file.endswith('.txt'):
            if txt_file == 'classes.txt':
                continue
            input_path = os.path.join(input_folder, txt_file)
            # im = cv2.imread(input_path[:-4]+'.jpg')
            # im_height, im_width, _ = im.shape
            im_height, im_width= 1400,1050
            output_path = os.path.join(output_folder, txt_file)
            with open(input_path, 'r') as file:
                lines = file.readlines()
            with open(output_path, 'w') as out_file:
                out_file.write('YOLO_OBB\n')
                for line in lines:
                    line = line.strip()
                    
                    params = list(map(float, line.split()))
                    class_id,x1, y1, x2, y2, x3, y3, x4, y4 = params
                    class_id, x_center, y_center, width, height, angle_deg = xyxyxyxy_to_xywhr_low_tolerance(class_id,x1, y1, x2, y2, x3, y3, x4, y4,im_height,im_width)
                    # out_file.write(str(class_id) + ' '+ str(x_center) +' '+ str(y_center) +' '+ str(width) +' '+ str(height) + ' '+ str(angle_deg) + '\n')
                    formatted_values = ["{:.6f}".format(value) for value in [x_center, y_center, width, height, angle_deg]]
                    output_line = "{} {}\n".format(str(int(class_id)), ' '.join(formatted_values))
                    out_file.write(output_line)
            # progress_retail = (index + 1) / total_fl * 100
            # progress_label.config(text=f"Converting YOLO OBB Dataset Format to DOTA Format: {progress_retail:.2f}%")
            # progress_label.update_idletasks()
    #         os.replace(output_path, input_path)
    # shutil.rmtree(output_folder)
des_path = r'C:\Users\CCSX009\Desktop\c1'
progress_label = 1   
format_params(des_path,progress_label)