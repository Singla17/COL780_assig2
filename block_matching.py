# -*- coding: utf-8 -*-
"""
COL780 Assigment-2

Somanshu Singla 2018EE10314
Lakshya Tangri  2018EE10222
"""

import cv2
import os
import numpy as np
from utils import get_bounding_box_dims
from utils import show_img
from utils import IOU

def matching_algo(inp_path,gt_path,ssd):
    
    image_list = os.listdir(inp_path)
    org_image = cv2.imread(os.path.join(inp_path,image_list[0]))
    gt_file = open(gt_path,'r')
    gt_file_content = gt_file.readlines()
    gt_box_dims = get_bounding_box_dims(gt_file_content, 1)
    template = org_image[gt_box_dims[1]:gt_box_dims[3],gt_box_dims[0]:gt_box_dims[2]]
    
    iou_sum = 0.0
    for i in range(1,len(image_list)):
        source = cv2.imread(os.path.join(inp_path,image_list[i]))
        if ssd:
            res = cv2.matchTemplate(source,template,cv2.TM_SQDIFF)
            val = np.amin(res)
            loc = np.where(res == val)
        else:
            res = cv2.matchTemplate(source,template,cv2.TM_CCOEFF_NORMED)
            val = np.amax(res)
            loc = np.where(res == val)
            
        box_pred = [0,0,0,0]
        box_pred[0]=loc[1][0]
        box_pred[1]=loc[0][0]
        box_pred[2]= loc[1][0]+(gt_box_dims[2]-gt_box_dims[0])
        box_pred[3]= loc[0][0]+(gt_box_dims[3]-gt_box_dims[1])
        box_gt= get_bounding_box_dims(gt_file_content, i+1)
        iou_sum += IOU(box_gt,box_pred)
        
        
        pred_img = cv2.rectangle(source,(box_pred[0],box_pred[1]),(box_pred[2],box_pred[3]),(255,0,0),2)
        if not os.path.isdir("Bolt/processed/"):
            os.mkdir("Bolt/processed/")
            cv2.imwrite("Bolt/processed/"+str(i)+".png",pred_img)
        else: 
            cv2.imwrite("Bolt/processed/"+str(i)+".png",pred_img)
        
        #show_img(pred_img)
    
    miou = iou_sum / (len(image_list)-1)
                 
    return miou
            
    
    
inp_path = "Bolt\img"
gt_path = "Bolt\groundtruth_rect.txt"
ssd = False
print(matching_algo(inp_path, gt_path, ssd))