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


def matching_algo(inp_path,gt_path,ssd,show,multiscale,adaptive,thresh):
    
    image_list = os.listdir(inp_path)
    org_image = cv2.imread(os.path.join(inp_path,image_list[0]))
    gt_file = open(gt_path,'r')
    gt_file_content = gt_file.readlines()
    gt_box_dims = get_bounding_box_dims(gt_file_content, 1)
    template = org_image[gt_box_dims[1]:gt_box_dims[3],gt_box_dims[0]:gt_box_dims[2]]
    
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    t_shape=template.shape
    iou_sum = 0.0
    
    for i in range(1,len(image_list)):

        frame = cv2.imread(os.path.join(inp_path,image_list[i]))
        frame = cv2.bilateralFilter(frame,15,75,75)
        rows, cols, ch = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gt_box_dims = get_bounding_box_dims(gt_file_content, i)
        box_pred = [0,0,0,0]
        
        """Initialization for affine"""
        # p = np.zeros([1,6], dtype = np.float64)
        W = np.array([[1,0,0],
                    [0, 1,0]],dtype = np.float64)
        # thresh=.025
        error=1
        itr=0
        errors=[]
        x=[]
        ssd=0
        while(error > thresh):
            """Warping """
            frame = cv2.warpAffine(frame, W, (frame.shape[1], frame.shape[0]))
            input_frame = frame[gt_box_dims[1]:gt_box_dims[3],gt_box_dims[0]:gt_box_dims[2]]
            """Compute Error"""
            # template = cv2.resize(template,(input_frame.shape[1],input_frame.shape[0]))
            diff = template - input_frame
            # ssd=np.sum(np.multiply(diff,diff))
            """Compute Warped Gradients"""
            gradX = cv2.Sobel(input_frame, cv2.CV_64F, 1, 0, ksize=5)
            gradY = cv2.Sobel(input_frame, cv2.CV_64F, 0, 1, ksize=5)
            
            
            ht, wt = template.shape
            Xc = np.tile(np.linspace(0, wt-1, wt), (ht, 1))
            Yc = np.tile(np.linspace(0, ht-1, ht), (wt, 1)).T
            
            
            """Compute Steepest Descent"""
            steepestDescentImages=[]
            steepestDescentImages.append(np.multiply(Xc,gradX))
            steepestDescentImages.append(np.multiply(Xc,gradY))
            steepestDescentImages.append(np.multiply(Yc,gradX))
            steepestDescentImages.append(np.multiply(Yc,gradY))
            steepestDescentImages.append(gradX)
            steepestDescentImages.append(gradY)
       
            
            
            
            """Compute Inverse Hessian"""
            l_2=[]
            for k in range(6):
                for j in range(6):
                    a=steepestDescentImages[k]
                    b=steepestDescentImages[j]
                    c=np.multiply(a,b)
                    l_2.append(np.sum(c))
            l_2=np.array(l_2)
            hess=l_2.reshape((6,6))
            
            l_3=[]
            for k in range(6):
                a=steepestDescentImages[k]
                c=np.multiply(a,diff)
                l_3.append(np.sum(c))
            l_3=np.array(l_3)
            
            
            if np.linalg.matrix_rank(hess) != 6:
                # print("hessian is singular")
                error=0
                continue
            """Compute delp by Multplying steepest Descent,Inverse Hessian,"""
            delp = np.matmul(np.linalg.inv(hess), l_3)
            """Update p"""
            error = np.linalg.norm(delp)
            if(error>0.5):
                break
            W= W + np.array([[delp[0],delp[2],delp[4]], [delp[1],delp[3],delp[5]]]) 
            
            errors.append(error)
            itr+=1
            x.append(itr)
        # print(len(errors))
        # print(len(x1))
        # plt.plot(x,errors)
        cornerPoints=np.array([[gt_box_dims[0],gt_box_dims[1]],[gt_box_dims[2],gt_box_dims[1]],[gt_box_dims[1],gt_box_dims[3]],[gt_box_dims[2],gt_box_dims[3]]])
        plotPts=[]
        for x,y in cornerPoints:
            pos=np.matmul(affine_Inv(W),[x,y,1])
            plotPts.append(pos)
        plotPts=np.array(plotPts)   
        plotPts = np.round(plotPts.astype(int))
        
        
        box_pred =[plotPts[0][0],plotPts[0][1],plotPts[-1][0],plotPts[-1][1]] 
        
        
                    
                
        source = cv2.imread(os.path.join(inp_path,image_list[i]))
        box_gt= get_bounding_box_dims(gt_file_content, i+1)
        iou_sum += IOU(box_gt,box_pred)
        
        # diff = template - input_frame
        # ssd=np.sum(np.multiply(diff,diff))
        # if(ssd < 0.1):
        # template = frame[box_pred[1]:box_pred[3],box_pred[0]:box_pred[2]]
            
            # template = cv2.resize(template, t_shape)
        # gt_box_dims=box_pred
        # print(gt_box_dims[0])
        if show:
            name = inp_path.split("\\")[0]
            pred_img = cv2.rectangle(source,(box_gt[0],box_gt[1]),(box_gt[2],box_gt[3]),(0,255,0),2)
            pred_img = cv2.rectangle(pred_img,(box_pred[0],box_pred[1]),(box_pred[2],box_pred[3]),(255,0,0),2)
            pred_img=cv2.rectangle(pred_img,(gt_box_dims[0],gt_box_dims[1]),(gt_box_dims[2],gt_box_dims[3]),(0,0,255),2)
            if not os.path.isdir(name+"/processed/"):
                os.mkdir(name+"/processed/")
                cv2.imwrite(name+"/processed/"+str(i)+".png",pred_img)
            else: 
                cv2.imwrite(name+"/processed/"+str(i)+".png",pred_img)
           
        
    
    miou = iou_sum / (len(image_list)-1)
                 
    return miou
            

test_class = "BlurCar2"   
inp_path = test_class+"\img"
gt_path = test_class+"\groundtruth_rect.txt"
ssd = False
show = True
multiscale = False
adaptive = True
thresh=0.025
print(matching_algo(inp_path, gt_path, ssd,show,multiscale,adaptive,thresh))
