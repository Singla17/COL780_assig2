# -*- coding: utf-8 -*-
"""
COL780 Assigment-2

Somanshu Singla 2018EE10314
Lakshya Tangri  2018EE10222
"""
import cv2
import numpy as np
def show_img(img):
    cv2.imshow("Mask",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def affine_Inv(prevWarp):
    R = prevWarp[:,0:2]
    rinv = np.linalg.inv(R)
    trans = np.matmul(rinv,prevWarp[:,2])
    pinv = np.array( [ [rinv[0,0], rinv[0,1] , -trans[0]],  [rinv[1,0], rinv[1,1] , -trans[1]]  ])
    return pinv
def get_bounding_box_dims(txt_file_content,line_no):
    """
    Inputs: txt_file- is the contents of the input file from which we are supposed to read broken line wise
            line_no- is the line_no which we are supposed to read

    Outputs: dims_array- stores the dimensions of a bounding box => form is [Tl corner X, TL corner Y, BR corner X, BR corner Y]
    """
    
    string_form = txt_file_content[line_no-1]
    try:
        dims_array = string_form.split(",")
        for i in range(len(dims_array)):
            dims_array[i]= int(dims_array[i])
    except:
        dims_array = string_form.split()
        for i in range(len(dims_array)):
            dims_array[i]= int(dims_array[i])
    
    dims_array[2]= dims_array[0]+dims_array[2]
    dims_array[3]= dims_array[3]+dims_array[1]
    return dims_array


def IOU(bbox_gt,bbox_pred):
    """
    Inputs: bbox_gt- Is the bounding box which represents the ground truth
            bbox_pred- Is the predicted bounding box
            
    Outputs: score- Is the IOU score for the given set of inputs
    
    Works only when edges of the two boxes are parellel
    """

    if bbox_gt[0]<bbox_pred[0] and bbox_gt[2]<bbox_pred[0]:
        return 0.0
    elif bbox_pred[0]<bbox_gt[0] and bbox_pred[2]<bbox_gt[0]:
        return 0.0
    elif bbox_gt[1]<bbox_pred[1] and bbox_gt[3]<bbox_pred[1]:
        return 0.0
    elif bbox_pred[1]<bbox_gt[1] and bbox_pred[3]<bbox_gt[1]:
        return 0.0
    
    hori_dist = []
    verti_dist = []
    hori_dist.append(bbox_gt[0])
    hori_dist.append(bbox_gt[2])
    hori_dist.append(bbox_pred[0])
    hori_dist.append(bbox_pred[2])
    verti_dist.append(bbox_gt[1])
    verti_dist.append(bbox_gt[3])
    verti_dist.append(bbox_pred[1])
    verti_dist.append(bbox_pred[3])
    
    hori_dist.sort()
    verti_dist.sort()
    h_diff = hori_dist[2]-hori_dist[1]
    v_diff = verti_dist[2]-verti_dist[1]
    intersection = (h_diff+1)*(v_diff+1)
    
    area_1 = (bbox_gt[2]-bbox_gt[0]+1)*(bbox_gt[3]-bbox_gt[1]+1)
    area_2 = (bbox_pred[2]-bbox_pred[0]+1)*(bbox_pred[3]-bbox_pred[1]+1)
    
    iou = intersection / float(area_1+area_2-intersection)
    
    return iou

"""
import os 
print(os.getcwd())    
file = open('Liquor\groundtruth_rect.txt')
cont = file.readlines()
oki = get_bounding_box_dims(cont, 1)
IOU([1,1,2,2],[2,2,4,4])
"""
