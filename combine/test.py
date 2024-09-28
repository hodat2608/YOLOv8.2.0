
import torch,os,cv2,shutil
import numpy as np
def xywhr2xyxyxyxy_original_ops(class_id,x,img_width,img_height):
        """
        Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4]. Rotation values should
        be in degrees from 0 to 90.

        Args:
            x (numpy.ndarray | torch.Tensor): Boxes in [cx, cy, w, h, rotation] format of shape (n, 5) or (b, n, 5).

        Returns:
            (numpy.ndarray | torch.Tensor): Converted corner points of shape (n, 4, 2) or (b, n, 4, 2).
        """
        cos, sin, cat, stack = (
            (torch.cos, torch.sin, torch.cat, torch.stack)
            if isinstance(x, torch.Tensor)
            else (np.cos, np.sin, np.concatenate, np.stack)
        )

        ctr = x[..., :2]
        w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
        cos_value, sin_value = cos(angle), sin(angle)
        vec1 = [w / 2 * cos_value, w / 2 * sin_value]
        vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
        vec1 = cat(vec1, -1)
        vec2 = cat(vec2, -1)
        pt1 = ctr + vec1 + vec2
        pt2 = ctr + vec1 - vec2
        pt3 = ctr - vec1 - vec2
        pt4 = ctr - vec1 + vec2

        corners = torch.stack([pt1, pt2, pt3, pt4], dim=-2)
        corners_normalized = corners.clone()
        corners_normalized[..., 0] = corners[..., 0] / img_width
        corners_normalized[..., 1] = corners[..., 1] / img_height 

        return [int(class_id)] + corners_normalized.view(-1).tolist()

def get_params_xywhr2xyxyxyxy_original_ops(des_path):
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
                        # class_id = params[0]
                        # bbox_list = params[1:]
                        # bbox_tensor = torch.tensor(bbox_list, dtype=torch.float32)
                        # bbox_tensor_2d = bbox_tensor.unsqueeze(0)
                        # converted_label = xywhr2xyxyxyxy_original_ops(class_id,bbox_tensor_2d,im_width,im_height)
                        # out_file.write(" ".join(map(str, converted_label)) + '\n')
                
                os.replace(output_path, input_path)
        shutil.rmtree(output_folder)
des_path = r'C:\Users\CCSX009\Videos\New folder'
get_params_xywhr2xyxyxyxy_original_ops(des_path)