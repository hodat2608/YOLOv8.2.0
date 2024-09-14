import numpy as np

angle = -15.110780
width = 540
height = 170
x_center = 758
y_center = 282
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
