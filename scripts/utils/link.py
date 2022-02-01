#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np

from pathlib import Path

FILE = Path(__file__).resolve()
ROS_WS = FILE.parents[3]  # 3단계 상위 폴더의 경로

class Link_Adabins:
    def __init__(self):
        # Insert Adabins's path 
        adabins_path = os.path.abspath(os.path.join(ROS_WS, 'adabins')) + "/scripts"
        sys.path.append(adabins_path) # utils 절대경로를 환경변수에 추가
        from utils_Adabins import Adabins
        self.adabins = Adabins(dataset='nyu')
        
    def predict(self, input_img):
        # image size
        if input_img.shape[0]!=480 or input_img.shape[1]!=640:
            input_img = input_img[:,160:1120,:] 
            input_img   = cv2.resize(input_img, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
            
            __, adabins_output = self.adabins.predict_pil(input_img)
            output = adabins_output.squeeze()
            
            output   = cv2.resize(output, dsize=(960, 720), interpolation=cv2.INTER_LINEAR)
            temp = np.zeros((720, 1280))
            temp[:,160:1120] = output

            output = temp
        else:
            __, adabins_output = self.adabins.predict_pil(input_img)
            output = adabins_output.squeeze()
            
        return output

class Link_DenseDepth:
    def __init__(self):
        # Insert DenseDepth's path 
        DenseDepth_path = os.path.abspath(os.path.join(ROS_WS, 'densedepth')) + "/scripts"
        sys.path.append(DenseDepth_path) # utils 절대경로를 환경변수에 추가
        from utils_DenseDepth import DenseDepth
        self.densedepth = DenseDepth()
        
    def predict(self, input_img):
        # image size
        if input_img.shape[0]!=480 or input_img.shape[1]!=640:
            input_img = input_img[:,160:1120,:] 
            input_img   = cv2.resize(input_img, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
            
            output = self.densedepth.predict(input_img)
            
            output   = cv2.resize(output, dsize=(960, 720), interpolation=cv2.INTER_LINEAR)
            temp = np.zeros((720, 1280))
            temp[:,160:1120] = output
            output = temp
        else:
            output = self.densedepth.predict(input_img)
            
        return output
    
class Link_DroNet:
    def __init__(self):
        # Insert DroNet's path 
        DroNet_path = os.path.abspath(os.path.join(ROS_WS, 'DroNet')) + "/scripts"
        sys.path.append(DroNet_path) # utils 절대경로를 환경변수에 추가
        from utils_DroNet import DroNet
        self.dronet = DroNet()
        
    def navigation(self, image):
        image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
        self.dronet.load_img(image)
        
        self.dronet.predict()
        
        # return steer, coll
    
    def get_vel(self):
        return self.dronet.get_vel()
    
    def get_steer(self):
        return self.dronet.get_steer()
        
    