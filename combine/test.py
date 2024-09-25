
import torch,os,cv2,shutil
import numpy as np
def xywhr2xyxyxyxy(class_id,x_center,y_center,width,height,angle,im_height,im_width):
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
        normalized_corners = final_corners / np.array([im_width,im_height])
        return [int(class_id)] + normalized_corners.flatten().tolist()

def get_params_xywhr2xyxyxyxy(des_path):
    input_folder = des_path
    os.makedirs(os.path.join(input_folder,'instance'),exist_ok=True)
    output_folder = (os.path.join(input_folder,'instance'))
    total_fl = len(des_path) 
    for index,txt_file in enumerate(os.listdir(input_folder)):
        if txt_file.endswith('.txt'):
            if txt_file == 'classes.txt':
                continue
            input_path = os.path.join(input_folder, txt_file)
            im = cv2.imread(input_path[:-4]+'.jpg')
            im_height, im_width, _ = im.shape
            output_path = os.path.join(output_folder, txt_file)
            with open(input_path, 'r') as file:
                lines = file.readlines()
            with open(output_path, 'w') as out_file:
                for line in lines:
                    line = line.strip()
                    if "YOLO_OBB" in line:
                        continue
                    params = list(map(float, line.split()))
                    class_id,x_center,y_center,width,height,angle = params
                    angle = abs(angle) if np.sign(angle) == -1 else 180-angle
                    print(class_id,x_center,y_center,width,height,angle)
                    converted_label = xywhr2xyxyxyxy(class_id,x_center,y_center,width,height,angle,im_height,im_width)
                    # print(converted_label)           
    #         os.replace(output_path, input_path)
    # shutil.rmtree(output_folder)
des_path = r'C:\Users\CCSX009\Desktop\compare\label_obb'
get_params_xywhr2xyxyxyxy(des_path)