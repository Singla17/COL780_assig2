# -*- coding: utf-8 -*-
"""
COL780 Assigment-2

Somanshu Singla 2018EE10314
Lakshya Tangri  2018EE10222
"""
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from utils import get_bounding_box_dims,IOU,affine_Inv


def matching_algo(inp_path,gt_path,ssd,show,multiscale,adaptive):
    
    image_list = os.listdir(inp_path)
    org_image = cv2.imread(os.path.join(inp_path,image_list[0]))
    gt_file = open(gt_path,'r')
    gt_file_content = gt_file.readlines()
    gt_box_dims = get_bounding_box_dims(gt_file_content, 1)
    template = org_image[gt_box_dims[1]:gt_box_dims[3],gt_box_dims[0]:gt_box_dims[2]]
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    iou_sum = 0.0
    
    for i in range(1,3):

        frame = cv2.imread(os.path.join(inp_path,image_list[i]))
        # frame = cv2.bilateralFilter(source,15,75,75)
        rows, cols, ch = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        box_pred = [0,0,0,0]
        
        """Initialization for affine"""
        p = np.zeros([1,6], dtype = np.float64)
        W = np.array([[1+p[0,0],p[0,2],p[0,4]],
                    [p[0,1], 1+p[0,3],p[0,5]]])
        thresh=.00005
        error=1
        itr=0
        errors=[]
        x=[]
        while(error > thresh):
            """Warping """
            frame = cv2.warpAffine(frame, W, (frame.shape[1], frame.shape[0]))
            input_frame = frame[gt_box_dims[1]:gt_box_dims[3],gt_box_dims[0]:gt_box_dims[2]]
            """Compute Error"""
            diff = template - input_frame
            """Compute Warped Gradients"""
            gradX = cv2.Sobel(input_frame, cv2.CV_64F, 1, 0, ksize=5)
            gradY = cv2.Sobel(input_frame, cv2.CV_64F, 0, 1, ksize=5)
            Height, Weight = template.shape
            Xc = np.tile(np.linspace(0, Weight-1, Weight), (Height, 1)).flatten()
            Yc = np.tile(np.linspace(0, Height-1, Height), (Weight, 1)).T.flatten()
            """Compute Steepest Descent"""
            steepest_descent = np.vstack([gradX.ravel() * Xc, gradY.ravel() * Xc,
                                          gradX.ravel()*Yc, gradY.ravel()*Yc, gradX.ravel(), gradY.ravel()]).T
            """Compute Inverse Hessian"""
            hessian = np.matmul(steepest_descent.T, steepest_descent)
            det_hessian = np.linalg.det(hessian)
            if det_hessian == 0:
                error=0
                continue
            """Compute delp by Multplying steepest Descent,Inverse Hessian,"""
            delp = np.matmul(np.linalg.inv(hessian), np.matmul(steepest_descent.T, diff.flatten()))
            """Update p"""
            W= W + np.array([[delp[0],delp[2],delp[4]], [delp[1],delp[3],delp[5]]]) 
            error = np.linalg.norm(delp)
            errors.append(error)
            itr+=1
            x.append(itr)
        print(len(errors))
        print(len(x))
        plt.plot(x,errors)
        cornerPoints=np.array([[gt_box_dims[0],gt_box_dims[1]],[gt_box_dims[2],gt_box_dims[1]],[gt_box_dims[1],gt_box_dims[3]],[gt_box_dims[2],gt_box_dims[3]]])
        tempPoints  = np.array([np.matmul(affine_Inv(W),[x,y,1]) for x,y in cornerPoints])
        tempPoints = np.round(tempPoints.astype(int))
        plotPoints = [i.astype(int) for i in tempPoints]
        pts = tempPoints.copy()
        pts = pts.reshape((-1,1,2))
        colorImage = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
        cv2.polylines(colorImage,[pts],True,(0,0,255))
        box_pred =[tempPoints[0][0],tempPoints[0][1],tempPoints[-1][0],tempPoints[-1][1]] 
        
        
                    
                
        source = cv2.imread(os.path.join(inp_path,image_list[i]))
        box_gt= get_bounding_box_dims(gt_file_content, i+1)
        iou_sum += IOU(box_gt,box_pred)
        
        
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
            

test_class = "A2\Bolt"   
inp_path = test_class+"\img"
gt_path = test_class+"\groundtruth_rect.txt"
ssd = False
show = True
multiscale = False
adaptive = True
print(matching_algo(inp_path, gt_path, ssd,show,multiscale,adaptive))
