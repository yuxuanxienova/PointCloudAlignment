import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

import numpy as np
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
def warp_ij(i_u,i_v,i_d,K,T_wi,T_wj):
    '''
    Input:
    i_u :[num_sample,1];x coordinate of point p projected on image_i viewed in image_i pixel frame 
    i_v :[num_sample,1];y coordinate of point p projected on image_i viewed in image_i pixel frame
    i_d :[num_sample,1];depth of point p viewed from camera_i frame
    K: [3,3] ; Camera intrinsic matrix
    T_wi: [4,4] ; Transformation matrix transform from ith camera frame to world frame
    T_wj: [4,4] ; Transformation matrix transform from jth camera frame to world frame

    output:
    j_u_ij : [num_sample,1];x coordinate of point p projected on image_j viewed in image_j pixel frame, obtained by warrping from camera_i frame
    j_v_ij : [num_sample,1];y coordinate of point p projected on image_j viewed in image_j pixel frame, obtained by warrping from camera_i frame
    j_d_ij : [num_sample,1];depth of point p viewed from camera_j frame, obtained by warrping from camera_i frame
    '''
    #process the input
    num_sample = i_u.shape[0]
    R_wi = T_wi[:3,:3]#[3,3]
    t_wi = T_wi[:3, 3].reshape((1,3,1)).repeat([num_sample,1,1])#[num_sample,3,1]
    R_wj = T_wj[:3,:3]#[3,3]
    t_wj = T_wj[:3, 3].reshape((1,3,1)).repeat([num_sample,1,1])#[num_sample,3,1]
    
    ones = torch.ones_like(i_u)#[num_sample,1]
    i_p=torch.stack([i_u,i_v,ones],dim=1)#[num_sample,3,1]
    
    #code the formula
    temp = torch.matmul(i_p,i_d.reshape((num_sample,1,1))) # [num_sample,3,1]
    K_inv = torch_inverse_3x3(K.reshape((1,3,3))).reshape((3,3)) # [3,3]
    cami_p = torch.matmul(K_inv , temp) # [num_sample,3,1]
    camj_p = torch.matmul(K , torch.matmul(R_wj.T , (torch.matmul(R_wi , cami_p) + t_wi - t_wj))).reshape((num_sample,3))#[num_sample,3]

    j_d_ij = camj_p[:,2]# [num_sample,1]
    j_u_ij = camj_p[:,0]/j_d_ij # [num_sample,1]
    j_v_ij = camj_p[:,1]/j_d_ij # [num_sample,1]

    return j_u_ij,j_v_ij,j_d_ij


if __name__ == "__main__":
    path_dataset = "./Data/Dataset1/"

    # 1. Load images and depth maps
    depth_mono=[]
    images = []
    transform = transforms.ToTensor()
    for img_id in range(10):
        depth_mono.append(transform(Image.open(path_dataset + 'depth_' + str(img_id) + '.jpg')))
        images.append(transform(Image.open(path_dataset + 'depth_' + str(img_id) + '.jpg')))

    # Stack the list of tensors along a new dimension
    depth_mono_stacked = torch.stack(depth_mono, dim=0)#[10,1,3024,4032]

    print(images[0].shape)
    H = images[0].shape[1]#3024
    W = images[0].shape[2]#4032

    u0 = W/2#2016
    v0 = H/2#1512

    num_img = 10#number of images
    num_iter = 5000#number of iterations
    #3. Initialize parameters

    alpha = torch.ones((num_img,1), requires_grad=True) #alpha[i] is global scale of ith image
    beta = torch.zeros((num_img,1),  requires_grad=True) #beta[i] is global shift of ith image

    alpha_local = torch.ones((num_img,H,W), requires_grad=True) #alpha[i][v][u] is local scale of ith image
    beta_local = torch.zeros((num_img,H,W), requires_grad=True) #beta[i][v][u] is local scale of ith image



    f = torch.tensor([1.2],requires_grad=True) # focal length, f=f_x=f_y
    pose_6dof = torch.zeros((num_img,6), requires_grad=True)# 6DoF parameters in the order of tx, ty, tz, rx, ry, rz; pose_6dof[i] represent T_wi(transform from world to i_th camera coordinate)
    # confidence = torch.ones((num_img,1),requires_grad=True)# sparse point weights


    K = torch.tensor([[f , 0. , u0],
                    [0. , f , v0],
                    [0. , 0. , 1. ]])

    optimizer = optim.Adam(params=[alpha, beta, alpha_local, beta_local,f, pose_6dof],lr=0.001)
    loss=[]
    #5.
    # Training loop
    num_epochs = 1
    for epoch in range(num_epochs):

        for iter in range(num_iter):
            
            #random sample pairs here!!
            i = random.randint(0,num_img-1)
            j = random.randint(0,num_img-1)
            if(i==j):continue
            #12 for paris i,j

            image_i = images[i].squeeze()
            image_j = images[j].squeeze()

            #13 compute scale consistent depth with Eq (3)
            depth_mono_i = depth_mono_stacked[i].squeeze()#[3024,4032]
            depth_mono_j = depth_mono_stacked[j].squeeze()#[3024,4032]

            #requires grad
            image_j.requires_grad=True
            image_i.requires_grad=True
            depth_mono_i.requires_grad=True
            depth_mono_j.requires_grad=True
            

            depth_i_global = depth_mono_i * alpha[i] + beta[i]#[3024,4032]
            depth_j_global = depth_mono_j * alpha[j] + beta[j]#[3024,4032]

            depth_i = alpha_local[i] * depth_i_global + beta_local[i]#[3024,4032]
            depth_j = alpha_local[j] * depth_j_global + beta_local[j]#[3024,4032]

            #14 compute T_ij
            T_wi = pose_vecToMat(pose_6dof[i].reshape(1,1,6))#[1,4,4]
            T_wj = pose_vecToMat(pose_6dof[j].reshape(1,1,6))#[1,4,4]
            T_iw = torch_inverse_T(T_wi)#[1,4,4]
            T_ij = torch.mm(T_iw.reshape(4,4),T_wj.reshape(4,4))#[1,4,4]

            L_gc = 0.

            #sample points


            # Generate coordinate grids
            x_coords, y_coords = np.meshgrid(np.arange(W), np.arange(H))

            # Flatten the coordinate grids to get 1D arrays of x and y coordinates
            x_coords_flat = x_coords.flatten()
            y_coords_flat = y_coords.flatten()

            # Randomly sample points from the image grid
            num_sample = 10000
            random_indices = np.random.choice(range(H * W), num_sample, replace=False)

            # Extract the sampled points from the image grid
            sampled_x_coords = x_coords_flat[random_indices]
            sampled_y_coords = y_coords_flat[random_indices]
            i_u_int = torch.tensor(sampled_x_coords).reshape((num_sample,1))#[num_sample,1]
            i_v_int = torch.tensor(sampled_y_coords).reshape((num_sample,1))#[num_sample,1]

            #convert to pytorch tensor in float data type, and reshape 
            i_u = torch.tensor(sampled_x_coords).reshape((num_sample,1)).float()#[num_sample,1]
            i_v = torch.tensor(sampled_y_coords).reshape((num_sample,1)).float()#[num_sample,1]

            #sample from depth map
            i_d = depth_i[i_v_int,i_u_int]#[num_sample,1]
            # print("i_u:{0},i_v:{0}".format(i_u,i_v))

            #apply warping
            j_u_ij,j_v_ij,j_d_ij = warp_ij(i_u=i_u,i_v=i_v,i_d=i_d,K=K,T_wi=T_wi.squeeze(),T_wj=T_wj.squeeze())
            # print("j_u_ij:{0},j_v_ij:{0}".format(j_u_ij,j_v_ij))

            # Drop points where either u or v is outside the desired range
            valid_indices = ((j_u_ij >= 0) & (j_u_ij < W-1) & (j_v_ij >= 0) & (j_v_ij < H-1)).squeeze()
            j_u_ij_valid = j_u_ij[valid_indices]#[num_sample_valid,1]
            j_v_ij_valid = j_v_ij[valid_indices]#[num_sample_valid,1]
            j_d_ij_valid = j_d_ij[valid_indices]#[num_sample_valid,1]
            # Round the tensors to the nearest integer
            j_u_ij_valid_rounded = torch.round(j_u_ij_valid).int()
            j_v_ij_valid_rounded = torch.round(j_v_ij_valid).int()


            j_d_valid = depth_j[j_v_ij_valid_rounded,j_u_ij_valid_rounded]

            #Loss Computation

            #Loss1: compute pixel wise photometric loss
            L_photo = torch.torch.sum(torch.abs(image_i[i_v_int,i_u_int] - image_j[j_v_ij_valid_rounded,j_u_ij_valid_rounded]))

            #Loss2: compute pixel wise geometric loss

            L_gc = torch.sum(torch.abs(j_d_ij_valid -j_d_valid )) 
            # L_rg = torch.sum(1-confidence)


            L = L_gc 
            loss.append(L.detach().numpy())
                    
            print(L.detach().numpy())
            optimizer.zero_grad()
            L.backward()
            optimizer.step()
                
                
    #visualization
    plt.plot(loss)
    plt.show()

