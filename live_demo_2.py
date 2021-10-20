# -*- coding: utf-8 -*-
"""
COL780 Assigment-2

Somanshu Singla 2018EE10314
Lakshya Tangri  2018EE10222
"""
import cv2
import numpy as np


def matching_algo(frame,gt_box_dims,thresh,pyr_len,itr_limit,temp_update,template,multiscale,ssd,adaptive):
    
    source = frame
    frame = cv2.bilateralFilter(frame,15,75,75)
    rows, cols, ch = frame.shape
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
    
    
                 
    return box_pred,template


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
                box_pred,template=matching_algo(frame,bbox,0.035,1,15,True,template,False,False,True)
                frame = cv2.rectangle(frame,(box_pred[0],box_pred[1]),(box_pred[2],box_pred[3]),(255,0,0),2)
            
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