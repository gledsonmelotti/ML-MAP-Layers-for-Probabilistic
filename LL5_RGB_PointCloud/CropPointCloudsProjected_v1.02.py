"From LL5"


import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from matplotlib.axes import Axes
from PIL import Image
from pyquaternion import Quaternion
from tqdm import tqdm

import scipy.io as sio

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import Box, RadarPointCloud, LidarPointCloud  # NOQA
from lyft_dataset_sdk.utils.geometry_utils import BoxVisibility, box_in_image, view_points  # NOQA
from lyft_dataset_sdk.utils.map_mask import MapMask

# Load the dataset
# Adjust the dataroot parameter below to point to your local dataset path.
# The correct dataset path contains at least the following four folders (or similar): images, lidar, maps, v1.0.1-train
level5data = LyftDataset(data_path='D:/Dataset_LyftLevel5/Perception/train', json_path='D:/Dataset_LyftLevel5/Perception/train/train_data', verbose=True)


#ALL = 0  # Requires all corners are inside the image.
#ANY = 1  # Requires at least one corner visible in the image.
#NONE = 2  # Requires no corners to be inside, i.e. box can be fully outside the image.


#######################
import pandas as pd
train = pd.read_csv('D:/Dataset_LyftLevel5/Perception/train/' + 'train.csv')
# Taken from https://www.kaggle.com/gaborfodor/eda-3d-object-detection-challenge

object_columns = ['sample_id', 'object_id', 'center_x', 'center_y', 'center_z',
                  'width', 'length', 'height', 'yaw', 'class_name']
objects = []
for sample_id, ps in tqdm(train.values[:]):
    object_params = ps.split()
    n_objects = len(object_params)
    for i in range(n_objects // 8):
        x, y, z, w, l, h, yaw, c = tuple(object_params[i * 8: (i + 1) * 8])
        objects.append([sample_id, i, x, y, z, w, l, h, yaw, c])
train_objects = pd.DataFrame(
    objects,
    columns = object_columns
)

numerical_cols = ['object_id', 'center_x', 'center_y', 'center_z', 'width', 'length', 'height', 'yaw']
train_objects[numerical_cols] = np.float32(train_objects[numerical_cols].values)
train_objects.head()
########################

#class LidarPointCloud(LidarPointCloud):
#    @staticmethod
#    def nbr_dims() -> int:
#        """Returns the number of dimensions.
#
#        Returns: Number of dimensions.
#
#        """
#        return 4
#
#    @classmethod
#    def from_file(cls, file_name: Path) -> "LidarPointCloud":
#        """Loads LIDAR data from binary numpy format. Data is stored as (x, y, z, intensity, ring index).
#
#        Args:
#            file_name: Path of the pointcloud file on disk.
#
#        Returns: LidarPointCloud instance (x, y, z, intensity).
#
#        """
#
#        assert file_name.suffix == ".bin", "Unsupported filetype {}".format(file_name)
#
#        scan = np.fromfile(str(file_name), dtype=np.float32)
#        if len(scan) % 5 == 0:
#            points = scan.reshape((-1, 5))[:, : cls.nbr_dims()]
#        else:
#            scan0=[]
#            scan1=[]
#            scan2=[]
#            scan3=[]
#            scan4=[]
#            scan_pontos = []
#            scan_pontos = scan.reshape((-1, 1))
#            dv = 0
#            for divisao in range(math.floor(len(scan)/5)):
#                if (dv+4)<=len(scan):
#                    scan0.append(scan_pontos[dv])
#                    scan1.append(scan_pontos[dv+1])
#                    scan2.append(scan_pontos[dv+2])
#                    scan3.append(scan_pontos[dv+3])
#                    scan4.append(scan_pontos[dv+4])
#                dv = dv+5
#            points = np.concatenate((scan0,scan1,scan2,scan3),axis=1)
#        return cls(points.T)


#def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
#    """This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
#    orthographic projections. It first applies the dot product between the points and the view. By convention,
#    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
#    normalization along the third dimension.
#
#    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
#    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
#    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
#     all zeros) and normalize=False
#
#    Args:
#        points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
#        view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
#        The projection should be such that the corners are projected onto the first 2 axis.
#        normalize: Whether to normalize the remaining coordinate (along the third axis).
#
#    Returns: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
#
#    """
#
#    assert view.shape[0] <= 4
#    assert view.shape[1] <= 4
#    assert points.shape[0] == 3
#
#    viewpad = np.eye(4)
#    viewpad[: view.shape[0], : view.shape[1]] = view
#
#    nbr_points = points.shape[1]
#
#    # Do operation in homogenous coordinates
#    points = np.concatenate((points, np.ones((1, nbr_points))))
#    points = np.dot(viewpad, points)
#    points = points[:3, :]
#
#    if normalize:
#        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)
#
#    return points

def LL5_map_pointcloud_to_image(pointsensor_token, camera_token):
    """Given a point sensor (lidar/radar) token and camera sample_data token, load point-cloud and map it to
    the image plane.

    Args:
        pointsensor_token: Lidar/radar sample_data token.
        camera_token: Camera sample_data token.

    Returns: (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).

    """

    cam = level5data.get("sample_data", camera_token)
    nome_cam=Path(cam["filename"]).parts
    nome_cam_part=nome_cam[1]
    nome_cam_size=len(nome_cam_part)
    nome_cam_final=nome_cam_part[0:nome_cam_size-5]
    pointsensor = level5data.get("sample_data", pointsensor_token)
    pcl_path = level5data.data_path / pointsensor["filename"]

    if pointsensor["sensor_modality"] == "lidar":
        pc = LidarPointCloud.from_file((pcl_path))
    else:
        pc = RadarPointCloud.from_file((pcl_path))
    image = Image.open(str(level5data.data_path / cam["filename"]))

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = level5data.get("calibrated_sensor", pointsensor["calibrated_sensor_token"])
    pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
    pc.translate(np.array(cs_record["translation"]))

    # Second step: transform to the global frame.
    poserecord = level5data.get("ego_pose", pointsensor["ego_pose_token"])
    pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
    pc.translate(np.array(poserecord["translation"]))

    # Third step: transform into the ego vehicle frame for the timestamp of the image.
    poserecord = level5data.get("ego_pose", cam["ego_pose_token"])
    pc.translate(-np.array(poserecord["translation"]))
    pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

    # Fourth step: transform into the camera.
    cs_record = level5data.get("calibrated_sensor", cam["calibrated_sensor_token"])
    pc.translate(-np.array(cs_record["translation"]))
    pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]

    # Retrieve the color from the depth.
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc.points[:3, :], np.array(cs_record["camera_intrinsic"]), normalize=True)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 0)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < image.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < image.size[1] - 1)    
    points = points[:, mask]
    coloring = coloring[mask]
    
    auxiliar = pc.points
    auxiliar = auxiliar[:, mask]
    auxiliar = np.delete(auxiliar, 0, 0)
    auxiliar = np.delete(auxiliar, 0, 0)
    points = np.delete(points, 2, 0)
    points = np.concatenate((points,auxiliar),axis=0)
    
    return points, coloring, image, nome_cam_final

#level5data.list_scenes()

for k in range(0,180):
    my_scene = level5data.scene[k]
    print(k)
    name = my_scene["name"]
    nbr_samples = my_scene['nbr_samples']
    for j in range(0,nbr_samples):
        if j==0:
            my_sample_token = my_scene["first_sample_token"]
            #level5data.render_sample(my_sample_token)
            my_sample = level5data.get('sample', my_sample_token)
            #my_sample
            #level5data.list_sample(my_sample['token'])
            codigo_token = my_sample['token']
        else:
            my_sample = level5data.get('sample', codigo_token)
            codigo_token = my_sample["next"]
            
        sample_token = codigo_token
        sample_record = level5data.get("sample", sample_token)
        pointsensor_channel = "LIDAR_TOP"
        #camera = ["CAM_FRONT_ZOOMED"]
        camera = ["CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT","CAM_FRONT_ZOOMED","CAM_BACK","CAM_BACK_LEFT","CAM_BACK_RIGHT"]
        # Here we just grab the front camera and the point sensor.
        for cam in range(0,len(camera)):
            camera_channel = camera[cam]
            pointsensor_token = sample_record["data"][pointsensor_channel]
            camera_token = sample_record["data"][camera_channel]
            points, coloring, image, nome_cam_final = LL5_map_pointcloud_to_image(pointsensor_token, camera_token)
            # Here we just grab the front camera and the point sensor.
    #        plt.figure(figsize=(9, 16))
    #        plt.imshow(im)
    #        plt.scatter(points[0, :], points[1, :], c=coloring, s=1)
    #        plt.axis("off")
            im = np.array(image)
#            if (len(im[:,0])==1080 and len(im[0,:])==1920):
#                name_final = str(k)+'_'+str(j)+'_'+name
#                sio.savemat('E:/LyfLevel5/DataSet/PointCloud_Image/'+name_final+'.mat',{'im':im})
#                sio.savemat('E:/LyfLevel5/DataSet/PointCloud_Projected/'+name_final+'.mat',{'points':points})
#                sio.savemat('E:/LyfLevel5/DataSet/PointCloud_Coloring/'+name_final+'.mat',{'coloring':coloring})

            #name_final = str(k)+'_'+str(j)+'_'+camera_channel
            sio.savemat('D:/Dataset_LyftLevel5/RangeView/PointCloud_Image/'+nome_cam_final+'.mat',{'im':im})
            sio.savemat('D:/Dataset_LyftLevel5/RangeView/PointCloud_Projected/'+nome_cam_final+'.mat',{'points':points})
            sio.savemat('D:/Dataset_LyftLevel5/RangeView/PointCloud_Coloring/'+nome_cam_final+'.mat',{'coloring':coloring})