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
from utils import IOU

def matching_algo(inp_path,gt_path,ssd,show,multiscale,adaptive):
    
    image_list = os.listdir(inp_path)
    org_image = cv2.imread(os.path.join(inp_path,image_list[0]))
    gt_file = open(gt_path,'r')
    gt_file_content = gt_file.readlines()
    gt_box_dims = get_bounding_box_dims(gt_file_content, 1)
    template = org_image[gt_box_dims[1]:gt_box_dims[3],gt_box_dims[0]:gt_box_dims[2]]
    
    iou_sum = 0.0
    
    for i in range(1,len(image_list)):

        source = cv2.imread(os.path.join(inp_path,image_list[i]))
        source = cv2.bilateralFilter(source,15,75,75)
        box_pred = [0,0,0,0]
        
        if not multiscale:
            if ssd:
                res = cv2.matchTemplate(source,template,cv2.TM_SQDIFF)
                val = np.amin(res)
                loc = np.where(res == val)
            else:
                res = cv2.matchTemplate(source,template,cv2.TM_CCOEFF_NORMED)
                val = np.amax(res)
                loc = np.where(res == val)
            box_pred[0]=loc[1][0]
            box_pred[1]=loc[0][0]
            box_pred[2]= loc[1][0]+(gt_box_dims[2]-gt_box_dims[0])
            box_pred[3]= loc[0][0]+(gt_box_dims[3]-gt_box_dims[1])
        
        else:
            source_height, source_width, _ = source.shape
            max_scaling = min(source_height/(gt_box_dims[3]-gt_box_dims[1]),source_width/(gt_box_dims[2]-gt_box_dims[0]))
            max_scaling = int(max_scaling)
            count = 1
            num_iters = 0
            max_val = 0
            final_loc = None
            chosen_iter = 0
            
            while max_scaling >=count:
                if ssd:
                    res = cv2.matchTemplate(source,template,cv2.TM_SQDIFF)
                    val = np.amin(res)
                    loc = np.where(res == val)
                else:
                    res = cv2.matchTemplate(source,template,cv2.TM_CCOEFF_NORMED)
                    val = np.amax(res)
                    loc = np.where(res == val)
                    
                count = count *2
                num_iters += 1
                source= cv2.pyrDown(source)
                if val > max_val:
                    final_loc = loc
                    chosen_iter = num_iters
            
            factor = 1
            for j in range(chosen_iter-1):
                factor = factor*2
                
            box_pred[0]=final_loc[1][0]
            box_pred[1]=final_loc[0][0]
            box_pred[2]= final_loc[1][0]+factor*(gt_box_dims[2]-gt_box_dims[0])
            box_pred[3]= final_loc[0][0]+factor*(gt_box_dims[3]-gt_box_dims[1])
                    
                
        source = cv2.imread(os.path.join(inp_path,image_list[i]))
        box_gt= get_bounding_box_dims(gt_file_content, i+1)
        iou_sum += IOU(box_gt,box_pred)
        
        if adaptive:
            if not multiscale:
                if not ssd:
                    if val >= 0.7:
                        template = source[box_pred[1]:box_pred[3],box_pred[0]:box_pred[2]]
                else:
                    if val < 0.1:
                        template = source[box_pred[1]:box_pred[3],box_pred[0]:box_pred[2]]
            else:
                if not ssd:
                    if max_val >= 0.7:
                        template = source[box_pred[1]:box_pred[3],box_pred[0]:box_pred[2]]
                else:
                    if max_val < 0.1:
                        template = source[box_pred[1]:box_pred[3],box_pred[0]:box_pred[2]]
        
        if show:
            name = inp_path.split("\\")[0]
            pred_img = cv2.rectangle(source,(box_gt[0],box_gt[1]),(box_gt[2],box_gt[3]),(0,255,0),2)
            pred_img = cv2.rectangle(pred_img,(box_pred[0],box_pred[1]),(box_pred[2],box_pred[3]),(255,0,0),2)
            if not os.path.isdir(name+"/processed/"):
                os.mkdir(name+"/processed/")
                cv2.imwrite(name+"/processed/"+str(i)+".png",pred_img)
            else: 
                cv2.imwrite(name+"/processed/"+str(i)+".png",pred_img)
        
    
    miou = iou_sum / (len(image_list)-1)
                 
    return miou
            
"""   
test_class = "Bolt"   
inp_path = test_class+"\img"
gt_path = test_class+"\groundtruth_rect.txt"
ssd = False
show = True
multiscale = False
adaptive = True
print(matching_algo(inp_path, gt_path, ssd,show,multiscale,adaptive))
"""
