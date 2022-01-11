#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np

from pathlib import Path

FILE = Path(__file__).resolve()
ROS_WS = FILE.parents[2]  # 2단계 상위 폴더의 경로


class Link_Adabins:
    def __init__(self):
        # Insert Adabins's path 
        adabins_path = os.path.abspath(os.path.join(ROS_WS, 'adabins')) + "/scripts"
        sys.path.append(adabins_path) # utils 절대경로를 환경변수에 추가
        from utils_Adabins import Adabins
        self.adabins = Adabins()
        
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

if __name__=="__main__":
    adabins = Link_Adabins()