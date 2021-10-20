# -*- coding: utf-8 -*-
"""
COL780 Assigment-2

Somanshu Singla 2018EE10314
Lakshya Tangri  2018EE10222
"""
import cv2
import numpy as np
from utils import affine_Inv

def getCorrectOrder(arr):
    
  if len(arr)<4:
      return arr
  
  cog=np.mean(arr,axis=0)
  sign=[np.sign(pt - cog) for pt in arr]
  map_u={}
  for i in  range(len(sign)):
    pt=sign[i]
    if((pt==np.array([-1.0,-1.0])).all()):
      map_u[0]=arr[i]
    elif ((pt==np.array([1.0,-1.0])).all()):
      map_u[1]=arr[i]
    elif ((pt==np.array([1.0,1.0])).all()):
      map_u[2]=arr[i]
    elif ((pt==np.array([-1.0,1.0])).all()):
      map_u[3]=arr[i]
  if len(map_u.keys())<4:
      return arr
  res=[map_u[0],map_u[1],map_u[2],map_u[3]]
  return np.array(res)

def getTransform(W,thresh,frame,template,gt_box_dims,itr_limit):
        error=1
        itr=0
        while(error > thresh and itr < itr_limit):
            
            """Warping """
            frame = cv2.warpAffine(frame, W, (frame.shape[1], frame.shape[0]))
            input_frame = frame[gt_box_dims[1]:gt_box_dims[3],gt_box_dims[0]:gt_box_dims[2]]
            
            """Compute Error"""
            if input_frame.shape != template.shape:
                height,width =input_frame.shape
                template = cv2.resize(template,(width,height))
            diff = template - input_frame
            
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
            itr+=1
       
        cornerPoints=np.array([[gt_box_dims[0],gt_box_dims[1]],[gt_box_dims[2],gt_box_dims[1]],[gt_box_dims[1],gt_box_dims[3]],[gt_box_dims[2],gt_box_dims[3]]])
        plotPts=[]
        for x,y in cornerPoints:
            pos=np.matmul(affine_Inv(W),[x,y,1])
            plotPts.append(pos)
        plotPts=np.array(plotPts)   
        plotPts = np.round(plotPts.astype(int))
        
        box_pred =[plotPts[0][0],plotPts[0][1],plotPts[-1][0],plotPts[-1][1]] 
        
        return W,box_pred,plotPts
        

def matching_algo(frame,gt_box_dims,thresh,pyr_len,itr_limit,temp_update,template):
    
    source = frame
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    frame = cv2.bilateralFilter(frame,15,75,75)
    rows, cols, ch = frame.shape
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    box_pred = [0,0,0,0]
 
    scaled_frames=[frame]
    scaled_templates=[template]
    scaled_gt_box_dims=[gt_box_dims]
    W = np.array([[1,0,0],
                [0, 1,0]],dtype = np.float64)
    
    for pyr in range(pyr_len):
        frame=cv2.pyrDown(frame)
        template=cv2.pyrDown(template)
        scaled_frames.append(frame)
        scaled_templates.append(template)
        temp=[]
        for l in range(2):
            temp.append((scaled_gt_box_dims[-1][l]+1)//2)
        temp_ht=(scaled_gt_box_dims[-1][2]-scaled_gt_box_dims[-1][0]+1)//2
        temp_width=(scaled_gt_box_dims[-1][3]-scaled_gt_box_dims[-1][1]+1)//2
        temp.append(temp[0]+temp_ht)
        temp.append(temp[1]+temp_width)
        scaled_gt_box_dims.append(temp)

      
        
    for idx in range(len(scaled_frames)-1,-1,-1):
        #print(idx)
        frame=scaled_frames[idx]
        template=scaled_templates[idx]
        gt_box_dims=scaled_gt_box_dims[idx]
        W,box_pred,plotPts=getTransform(W,thresh,frame,template,gt_box_dims,itr_limit)
        
    
    if temp_update:
        template = source[box_pred[1]:box_pred[3],box_pred[0]:box_pred[2]]
                 
    return plotPts,template,box_pred


####################  Video Capture ##########################

if __name__ == "__main__":

    vid = cv2.VideoCapture(0)
    if not (vid.isOpened()):
        print("Unable to start camera")
    
    num_frame = 1
    ret, frame = vid.read()
    bbox=list(cv2.selectROI("frame",frame,False))
    bbox[2]= bbox[0]+bbox[2]
    bbox[3]= bbox[1]+bbox[3]
    template = frame[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    
    while (True):
        
        ret, frame = vid.read()
            
        if ret == True:
            try: 
                bounding_polygon,template,bbox=matching_algo(frame,bbox,0.035,1,15,True,template)
                bounding_polygon = getCorrectOrder(bounding_polygon)
                bounding_polygon = bounding_polygon.reshape((-1,1,2))
                frame = cv2.polylines(frame,[bounding_polygon],True,(255,0,0),1)
            except:
                print("Some error occured")
                break
                
            
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):         ## GfG reference
                break
        else:
            break
        
        num_frame += 1
        
    vid.release()
    cv2.destroyAllWindows()