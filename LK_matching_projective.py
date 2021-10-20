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
from utils import get_bounding_box_dims,IOU,affine_Inv,projective_Inv

def getWarp(p):
    return np.array([[p[0],p[1],p[2]],
                    [p[3], p[4],p[5]],[p[6],p[7],1]],dtype = np.float64)
def matching_algo(inp_path,gt_path,ssd,show,multiscale,adaptive,thresh,itr_limit,temp_update):
    
    image_list = os.listdir(inp_path)
    org_image = cv2.imread(os.path.join(inp_path,image_list[0]))
    gt_file = open(gt_path,'r')
    gt_file_content = gt_file.readlines()
    gt_box_dims = get_bounding_box_dims(gt_file_content, 1)
    template = org_image[gt_box_dims[1]:gt_box_dims[3],gt_box_dims[0]:gt_box_dims[2]]
    
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    iou_sum = 0.0
    
    for i in range(1,len(image_list)):

        frame = cv2.imread(os.path.join(inp_path,image_list[i]))
        frame = cv2.bilateralFilter(frame,15,75,75)
        rows, cols, ch = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gt_box_dims = get_bounding_box_dims(gt_file_content, i)
        box_pred = [0,0,0,0]
        
        """Initialization for projective"""
        p=np.array([1,0,0,0,1,0,0,0],dtype=np.float64)
        W = getWarp(p)
       
        error=1
        itr=0
        errors=[]
        x=[]
        while(error > thresh and itr <itr_limit):
            """Warping """
            frame = cv2.warpPerspective(frame, W, (frame.shape[1], frame.shape[0]))
            input_frame = frame[gt_box_dims[1]:gt_box_dims[3],gt_box_dims[0]:gt_box_dims[2]]
            """Compute Error"""
            diff = template - input_frame
            
            """Compute Warped Gradients"""
            gradX = cv2.Sobel(input_frame, cv2.CV_64F, 1, 0, ksize=5)
            gradY = cv2.Sobel(input_frame, cv2.CV_64F, 0, 1, ksize=5)
            
            
            ht, wt = template.shape
            Xc = np.tile(np.linspace(0, wt-1, wt), (ht, 1))
            Yc = np.tile(np.linspace(0, ht-1, ht), (wt, 1)).T
            
            
            """Compute Steepest Descent"""
            denom=p[6]*Xc+p[7]*Yc+1
            x_num=p[0]*Xc+p[1]*Yc+p[2]
            y_num=p[3]*Xc+p[4]*Yc+p[5]
            steepestDescentImages=[]
            steepestDescentImages.append((Xc*gradX)/denom)
            steepestDescentImages.append((Yc*gradX)/denom)
            steepestDescentImages.append((gradX)/denom)
            
            steepestDescentImages.append((Xc*gradY)/denom)
            steepestDescentImages.append((Yc*gradY)/denom)
            steepestDescentImages.append((gradY)/denom)
            
            steepestDescentImages.append((-1*(Xc*gradX*x_num)/denom**2) + ((gradY*Xc*y_num*-1)/(denom**2)))
            steepestDescentImages.append((-1*(Yc*gradX*x_num)/denom**2) + ((gradY*Yc*y_num*-1)/(denom**2)))
            # steepestDescentImages.append((-1*(gradX*x_num)/denom**2) + ((gradY*y_num*-1)/(denom**2)))
            
            
            
           
            
       
            
            
            
            """Compute Inverse Hessian"""
            l_2=[]
            for k in range(8):
                for j in range(8):
                    a=steepestDescentImages[k]
                    b=steepestDescentImages[j]
                    c=np.multiply(a,b)
                    l_2.append(np.sum(c))
            l_2=np.array(l_2)
            hess=l_2.reshape((8,8))
            
            l_3=[]
            for k in range(8):
                a=steepestDescentImages[k]
                c=np.multiply(a,diff)
                l_3.append(np.sum(c))
            l_3=np.array(l_3)
            # print(np.linalg.matrix_rank(hess))
            
            if np.linalg.matrix_rank(hess) != 8:
                # print("hessian is singular")
                error=0
                continue
            """Compute delp by Multplying steepest Descent,Inverse Hessian,"""
            delp = np.matmul(np.linalg.inv(hess), l_3)
            """Update p"""
            p+=delp
            error = np.linalg.norm(delp)
            if(error>0.5):
                break
            delW=getWarp(delp)
            W+= delW
            # print(delW)
            errors.append(error)
            itr+=1
            x.append(itr)
            
        
        # print(len(errors))
        # print(len(x1))
        # plt.plot(x,errors)
        cornerPoints=np.array([[gt_box_dims[0],gt_box_dims[1]],[gt_box_dims[2],gt_box_dims[1]],[gt_box_dims[1],gt_box_dims[3]],[gt_box_dims[2],gt_box_dims[3]]])
        plotPts=[]
        for x,y in cornerPoints:
            pos=np.matmul(projective_Inv(W),[x,y,1])
            plotPts.append(pos)
        plotPts=np.array(plotPts)   
        plotPts = np.round(plotPts.astype(int))
        
        
        box_pred =[plotPts[0][0],plotPts[0][1],plotPts[-1][0],plotPts[-1][1]] 
        
        
                    
                
        source = cv2.imread(os.path.join(inp_path,image_list[i]))
        box_gt= get_bounding_box_dims(gt_file_content, i+1)
        iou_sum += IOU(box_gt,box_pred)

        if temp_update:
            template = frame[box_pred[1]:box_pred[3],box_pred[0]:box_pred[2]]
            gt_box_dims=box_pred
        
        
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
            

test_class = "Bolt"   
inp_path = test_class+"\img"
gt_path = test_class+"\groundtruth_rect.txt"
ssd = False
show = True
multiscale = False
adaptive = True
thresh=0.035
itr_lim = 15
temp_update = False
print(matching_algo(inp_path, gt_path, ssd,show,multiscale,adaptive,thresh,itr_lim,temp_update))
