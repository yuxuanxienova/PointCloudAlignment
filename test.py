
#--------------------------------------------test1 use trimesh----------------------------------------
# import trimesh
# import numpy as np

# # Generate random point cloud data with colors
# num_points = 100
# point_cloud = np.random.rand(num_points, 3)  # Random point cloud with 100 points
# colors = np.random.rand(num_points, 3)  # Random colors for each point

# # Create a trimesh.PointCloud object with vertices and colors
# pc_mesh = trimesh.points.PointCloud(vertices=point_cloud, colors=colors)

# # Create a scene with the point cloud
# scene = trimesh.Scene([pc_mesh])

# # Show the scene
# scene.show()

#--------------------------------------------test2 load dapth map----------------------------------
cam_cen_u = 651.213
cam_cen_v = 845.885
cam_f_u = 1389.988
cam_f_v = 1390.1715

import cv2
import numpy as np
import matplotlib.pyplot as plt

import os

file_path = r'./Data/TestData/depth_1.png'

if os.path.exists(file_path):
    print("File path exists.")
else:
    print("File path does not exist.")

# Load the raw depth map
depth_map = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

# # Convert the depth map to float32 and normalize it
# depth_map_float = depth_map.astype(np.float32) / 65535.0

# Visualize the depth map
plt.figure(figsize=(8, 6))
plt.imshow(depth_map, cmap='jet')
plt.colorbar(label='Depth (normalized)')
plt.title('Depth Map')
plt.xlabel('Pixel')
plt.ylabel('Pixel')
plt.show()