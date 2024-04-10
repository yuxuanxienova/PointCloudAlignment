
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
# cam_cen_u = 651.213
# cam_cen_v = 845.885
# cam_f_u = 1389.988
# cam_f_v = 1390.1715

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# import os

# file_path = r'./Data/TestData/depth_1.png'

# if os.path.exists(file_path):
#     print("File path exists.")
# else:
#     print("File path does not exist.")

# # Load the raw depth map
# depth_map = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

# # # Convert the depth map to float32 and normalize it
# # depth_map_float = depth_map.astype(np.float32) / 65535.0

# # Visualize the depth map
# plt.figure(figsize=(8, 6))
# plt.imshow(depth_map, cmap='jet')
# plt.colorbar(label='Depth (normalized)')
# plt.title('Depth Map')
# plt.xlabel('Pixel')
# plt.ylabel('Pixel')
# plt.show()
#-----------------------------------test3 monocular depth--------------------------------------- 
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# import cv2
# import torch
# import urllib.request

# import matplotlib.pyplot as plt

# url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# urllib.request.urlretrieve(url, filename)

# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# midas = torch.hub.load("intel-isl/MiDaS", model_type)
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# midas.to(device)
# midas.eval()

# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

# if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
#     transform = midas_transforms.dpt_transform
# else:
#     transform = midas_transforms.small_transform
# img = cv2.imread(filename)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# input_batch = transform(img).to(device)

# with torch.no_grad():
#     prediction = midas(input_batch)

#     prediction = torch.nn.functional.interpolate(
#         prediction.unsqueeze(1),
#         size=img.shape[:2],
#         mode="bicubic",
#         align_corners=False,
#     ).squeeze()

# output = prediction.cpu().numpy()

# # Display the depth map
# plt.imshow(output, cmap='inferno')  # You can choose any colormap you prefer
# plt.colorbar()  # Add a color bar to show the depth scale
# plt.axis('off')  # Turn off axis
# plt.title('Depth Map')
# plt.show()

#-----------------------------test transoformation-------------------------

import os
import torch
from PIL import Image
from torchvision import transforms
def torch_det_2x2(tensor):
    '''
    tensor: torch.tensor, [B, 2, 2]
    '''

    a = tensor[:, 0, 0]
    b = tensor[:, 0, 1]
    c = tensor[:, 1, 0]
    d = tensor[:, 1, 1]

    return a * d - b * c

def torch_inverse_3x3(tensor):
    '''
    tensor: torch.tensor, [B, 3, 3]
    [ a1, b1, c1
      a2, b2, c2
      a3, b3, c3 ]
    '''

    a1 = tensor[:, 0, 0]; b1 = tensor[:, 0, 1]; c1 = tensor[:, 0, 2]
    a2 = tensor[:, 1, 0]; b2 = tensor[:, 1, 1]; c2 = tensor[:, 1, 2]
    a3 = tensor[:, 2, 0]; b3 = tensor[:, 2, 1]; c3 = tensor[:, 2, 2]

    coefficient = 1 / (a1 * (b2 * c3 - c2 * b3) - a2 * (b1 * c3 - c1 * b3) + a3 * (b1 * c2 - c1 * b2))

    tensor_new = torch.stack([ 
        (b2 * c3 - c2 * b3), (c1 * b3 - b1 * c3), (b1 * c2 - c1 * b2),
        (c2 * a3 - a2 * c3), (a1 * c3 - c1 * a3), (a2 * c1 - a1 * c2),
        (a2 * b3 - b2 * a3), (b1 * a3 - a1 * b3), (a1 * b2 - a2 * b1),
    ])
    tensor_new = tensor_new.permute(1, 0).view(-1, 3, 3)
    tensor_new = tensor_new * coefficient[:, None, None]

    return tensor_new

def torch_inverse_T(T):
    '''
    T: torch.tensor, [B,4,4]

    T_inv : torch.tensor, [B,4,4]
    '''
    B = T.shape[0]
    R = T[:,:3,:3]#[B,3,3]
    t = T[:,:3, 3].reshape(B,3,1)#[B,3,1]

    R_inv = torch_inverse_3x3(R) # [B,3,3]
    t_inv = -t # [B,3,1]

    temp = torch.cat([R_inv,t_inv],dim=2)#[B,3,4]
    temp2 = torch.tensor([0,0,0,1]).repeat([B,1,1])#[B,1,4]
    T_inv = torch.cat([temp,temp2],dim=1)#[B,4,4]
    return T_inv


def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat
def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:, :1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:,
                                            1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

def pose_vecToMat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 1 ,6]
    Returns:
        A T matrix -- [B, 4, 4]
    """
    B = vec.shape[0]


    translation = vec[:, :,:3].reshape((B,3,1)) # [B, 3, 1]
    rot = vec[:, :,3:] # [B,1,3]
    
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot.reshape(B,3))  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]


    T = torch.cat([transform_mat,torch.tensor([0,0,0,1]).repeat([B,1,1])],dim=1)
    return T

vec = torch.zeros((1,6))
vec = torch.Tensor([1,1,2,0,0,0]).repeat([5,1,1])
T = pose_vecToMat(vec)
print("T:{0}".format(T))

T_inv = torch_inverse_T(T)
print("T_inv:{0}".format(T_inv))

