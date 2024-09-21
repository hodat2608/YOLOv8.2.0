import numpy as np

angle = 89.608093
width = 258.318475
height = 90.997871
x_center = 216.381323
y_center = 309.154998
# Chuyển góc từ độ sang radian
angle_rad = np.deg2rad(angle)
print('angle_rad',angle_rad)
# Tạo ma trận quay dựa trên góc nghiêng
rotation_matrix = np.array([
    [np.cos(angle_rad), -np.sin(angle_rad)],
    [np.sin(angle_rad), np.cos(angle_rad)]
])
print('rotation_matrix',rotation_matrix)
half_width = width / 2
half_height = height / 2

corners = np.array([
    [-half_width, -half_height],  # Bottom-left
    [half_width, -half_height],   # Bottom-right
    [half_width, half_height],    # Top-right
    [-half_width, half_height]    # Top-left
])
print('corners',corners)
rotated_corners = np.dot(corners, rotation_matrix)
print('rotated_corners',rotated_corners)
final_corners = rotated_corners + np.array([x_center, y_center])
print('final_corners',final_corners)
normalized_corners = final_corners / np.array([1200, 1200])
print('normalized_corners',normalized_corners)
a = normalized_corners.flatten().tolist()
# a = " ".join(map(str, a)) + '\n'
print(a)



# import numpy as np
# import math

# def xyxyxyxy_to_xywhr(x1, y1, x2, y2, x3, y3, x4, y4, img_width, img_height):
#     """
#     Chuyển đổi từ định dạng xyxyxyxy (tọa độ bốn đỉnh) sang xywhr (trung tâm, kích thước, góc quay).
    
#     Parameters:
#     - x1, y1, x2, y2, x3, y3, x4, y4: Tọa độ bốn đỉnh của bounding box (giá trị chuẩn hóa từ 0 đến 1).
#     - img_width: Chiều rộng của ảnh (pixel).
#     - img_height: Chiều cao của ảnh (pixel).
    
#     Returns:
#     - x_center: Tọa độ trung tâm theo trục x (pixel).
#     - y_center: Tọa độ trung tâm theo trục y (pixel).
#     - width: Chiều rộng của bounding box (pixel).
#     - height: Chiều cao của bounding box (pixel).
#     - angle_rad: Góc quay của bounding box so với trục x (radian).
#     """
#     x_center_norm = (x1 + x2 + x3 + x4) / 4
#     y_center_norm = (y1 + y2 + y3 + y4) / 4
#     width_norm = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
#     height_norm = np.sqrt((x4 - x1)**2 + (y4 - y1)**2)
#     angle_rad = np.arctan2(y2 - y1, x2 - x1)
#     angle_deg = np.degrees(angle_rad)
#     x_center = x_center_norm * img_width
#     y_center = y_center_norm * img_height
#     width = width_norm * img_height
#     height = height_norm * img_width
#     return x_center, y_center, width, height, angle_deg

# # Ví dụ sử dụng
# # # Các tọa độ của 4 đỉnh của bounding box sau khi được chuẩn hóa (0 đến 1)
# # x1, y1 = 0.12142857203505845, 0.41714285780207405
# # x2, y2 = 0.12269064491402601, 0.1711310176955486
# # x3, y3 = 0.18768760367922727, 0.17172380505506873
# # x4, y4 = 0.18642553080025973, 0.4177356451615942


# x1, y1 = 0.8350346878325432, 0.16923538316670492
# x2, y2 =  0.8362305805042252, 0.4152478110262435
# x3, y3 = 0.771233466453171, 0.41580951397615223
# x4, y4 =   0.7700375737814891, 0.16979708611661362

# # Kích thước ảnh
# img_width = 1400  # pixel
# img_height = 1050  # pixel

# # Chuyển đổi sang định dạng xywhr
# x_center, y_center, width, height, angle_rad = xyxyxyxy_to_xywhr(
#     x1, y1, x2, y2, x3, y3, x4, y4, img_width, img_height
# )

# print(f"x_center: {x_center}")
# print(f"y_center: {y_center}")
# print(f"width: {width}")
# print(f"height: {height}")
# print(f"angle_rad: {angle_rad}")
