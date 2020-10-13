import os
import gc
import numpy as np
import pandas as pd

import json
import math
import sys
import time
from datetime import datetime
from typing import Tuple, List

import cv2
import matplotlib.pyplot as plt
from collections import OrderedDict
from shapely.geometry import MultiPoint, box
from pyquaternion import Quaternion
from tqdm import tqdm


from lyft_dataset_sdk.utils.data_classes import Box, LidarPointCloud, RadarPointCloud  # NOQA
from lyft_dataset_sdk.utils.map_mask import MapMask
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from lyft_dataset_sdk.utils.geometry_utils import transform_matrix

from pathlib import Path

import struct
from abc import ABC, abstractmethod
from functools import reduce
from typing import Tuple, List, Dict
import copy
import scipy.io as sio

###############################################################################
# box_vis_level: BoxVisibility = BoxVisibility.ANY  # Requires at least one corner visible in the image.
box_vis_level: BoxVisibility = BoxVisibility.ALL  # Requires all corners are inside the image.

def post_process_coords(corner_coords, w, h):
    imsize=(w,h)
    """
    Get the intersection of the convex hull of the reprojected bbox corners and the image canvas, return None if no
    intersection.
    :param corner_coords: Corner coordinates of reprojected bounding box.
    :param imsize: Size of the image canvas.
    :return: Intersection of the convex hull of the 2D box corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return []

def gledson_render_annotation(my_annotation_token, w, h, view = np.eye(4),box_vis_level=box_vis_level):
    ann_token=my_annotation_token
    w=w
    h=h
    """Render selected annotation.

    Args:
        ann_token: Sample_annotation token.
        margin: How many meters in each direction to include in LIDAR view.
        view: LIDAR view point.
        box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        out_path: Optional path to save the rendered figure to disk.

    """

    ann_record = lyftd.get("sample_annotation", ann_token)
    categoria=ann_record['category_name']
    sample_record = lyftd.get("sample", ann_record["sample_token"])
    lidar_top = [key for key in sample_record["data"].keys() if "LIDAR_TOP" in key]
    if lidar_top==['LIDAR_TOP']:
        boxes_cam, cam, boxes_lidar = [], [], []
        cams = [key for key in sample_record["data"].keys() if "CAM" in key]
        camera=[]
        for cam in cams:
            data_path_cam, boxes_cam, camera_intrinsic = lyftd.get_sample_data(
                sample_record["data"][cam], box_vis_level=box_vis_level, selected_anntokens=[ann_token])
            if len(boxes_cam) > 0:
                break  # We found an image that matches. Let's abort.
        if len(boxes_cam)==1:
            camera=cam
            cam = sample_record["data"][cam]
                    # CAMERA view
            path_cam, boxes_cam, camera_intrinsic = lyftd.get_sample_data(cam, selected_anntokens=[ann_token])
            data_path_cam=path_cam.parts[5]
            corners_camera3D=[]
            corners_camera2D=[]
            for f in range(len(boxes_cam)):
                 corners_camera3D=boxes_cam[f].corners().T
                
                 corners_3d = boxes_cam[f].corners()#corners_3d[:, in_front]
    
                # Project 3d box to 2d.
                 corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()
    
                # Keep only corners that fall within the image.
                 final_coords = post_process_coords(corner_coords, w, h)
                 #min_x, min_y, max_x, max_y = final_coords
                 corners_camera2D=final_coords
                 
            # LIDAR view
            lidar = sample_record["data"]["LIDAR_TOP"]
            path_lidar, boxes_lidar, lidar_intrinsic = lyftd.get_sample_data(lidar, selected_anntokens=[ann_token])
            data_path_lidar=path_lidar.parts[5]
            pc=LidarPointCloud.from_file(path_lidar)
            pointclouds = pc.points.T
            corners_lidar=[]
            #for box in boxes_lidar:
            for v in range(len(boxes_lidar)):
                boxes3D=boxes_lidar[v].corners()
                corners_lidar=boxes3D.T
    
            return data_path_cam, camera, corners_camera3D, corners_camera2D, data_path_lidar, corners_lidar, pointclouds, categoria
        else:
            b='boxes_cam vazio'
            print(b)
            data_path_cam=[]
            camera=[] 
            corners_camera3D=[]
            corners_camera2D=[]
            data_path_lidar=[]
            corners_lidar=[]
            pointclouds=[]
            categoria=[]
            return data_path_cam, camera, corners_camera3D, corners_camera2D, data_path_lidar, corners_lidar, pointclouds, categoria
    else:
         a='lidar_top é diferente de LIDAR_TOP'
         print(a)

DATA_PATH = 'D:/Dataset_LyftLevel5/Perception/train/'

lyftd = LyftDataset(data_path=DATA_PATH, json_path=DATA_PATH+'train_data')
lyftd.category
lyftd.attribute
attribute=[]
attribute1=lyftd.attribute
for u in range(len(attribute1)):
    attribute.append(attribute1[u]['token'])

total_scene = lyftd.scene

data_path_cam_final=[]
camera_final=[]
boxes_cam_final3D=[]
camera_intrinsic_final=[]
data_path_lidar_final=[]
boxes_lidar_final=[]
categoria_final=[]
pointclouds_final=[]
boxes_cam_final2D=[]

j=116
for jj in range(len(total_scene)):
    print('Impar:', j)
    my_scene = lyftd.scene[j]
    my_sample_token_first = my_scene["first_sample_token"]
    my_sample_first = lyftd.get('sample', my_sample_token_first)
    lidar_top_first = my_sample_first['data']['LIDAR_TOP']
    first_token=my_sample_token_first
    my_sample_token_last = my_scene["last_sample_token"]
    my_sample_last = lyftd.get('sample', my_sample_token_last)
    lidar_top_last = my_sample_last['data']['LIDAR_TOP']
    last_token=my_sample_token_last
    next_token=my_sample_token_first
    nexttoken=[]
    while next_token!='':
      nexttoken.append(next_token)
      next_token=lyftd.get('sample', next_token)['next']
    next_flip=nexttoken
    first_next_last_token= nexttoken
    for v in range(len(first_next_last_token)):
        sensor_channel = 'CAM_FRONT_ZOOMED' # usado para obter apenas o tamanho das imagens
        my_sample_gledson = lyftd.get('sample', first_next_last_token[v])
        my_sample_gledson_data = lyftd.get('sample_data', my_sample_gledson['data'][sensor_channel])
        w=my_sample_gledson_data['width']
        h=my_sample_gledson_data['height']
        for k in range(len(my_sample_gledson['anns'])):
            #print('anns:', k) # Cena, amostra da cena e dados da cena
            print(str(j)+'_'+str(v)+'_'+str(k))
            my_annotation_token = my_sample_gledson['anns'][k]
            box_vis_level: BoxVisibility = BoxVisibility.ALL
            data_path_cam, camera, corners_camera3D, corners_camera2D, data_path_lidar, corners_lidar, pointclouds, categoria = gledson_render_annotation(my_annotation_token, w, h, view = np.eye(4), box_vis_level=box_vis_level)
            if len(data_path_cam)!=0 and len(camera)!=0 and len(corners_camera3D)!=0 and len(corners_camera2D)!=0 and len(data_path_lidar)!=0 and len(corners_lidar)!=0 and len(pointclouds)!=0 and len(categoria)!=0:
                if w==2048 and h==864:
                    sio.savemat('D:/Dataset_LyftLevel5/2048_par/data_path_cam/'+str(j)+'_'+str(v)+'_'+str(k)+'.mat',{'data_path_cam':data_path_cam})
                    sio.savemat('D:/Dataset_LyftLevel5/2048_par/camera/'+str(j)+'_'+str(v)+'_'+str(k)+'.mat',{'camera':camera})
                    sio.savemat('D:/Dataset_LyftLevel5/2048_par/corners_camera3D/'+str(j)+'_'+str(v)+'_'+str(k)+'.mat',{'corners_camera3D':corners_camera3D})
                    sio.savemat('D:/Dataset_LyftLevel5/2048_par/corners_camera2D/'+str(j)+'_'+str(v)+'_'+str(k)+'.mat',{'corners_camera2D':corners_camera2D})
                    sio.savemat('D:/Dataset_LyftLevel5/2048_par/data_path_lidar/'+str(j)+'_'+str(v)+'_'+str(k)+'.mat',{'data_path_lidar':data_path_lidar})
                    sio.savemat('D:/Dataset_LyftLevel5/2048_par/corners_lidar/'+str(j)+'_'+str(v)+'_'+str(k)+'.mat',{'corners_lidar':corners_lidar})
                    sio.savemat('D:/Dataset_LyftLevel5/2048_par/pointclouds/'+str(j)+'_'+str(v)+'_'+str(k)+'.mat',{'pointclouds':pointclouds})
                    sio.savemat('D:/Dataset_LyftLevel5/2048_par/categoria/'+str(j)+'_'+str(v)+'_'+str(k)+'.mat',{'categoria':categoria})
    j=j+2