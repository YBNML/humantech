#!/usr/bin/env python3

'''
< 00_our_Navi.py >
1. Monocular Depth Estimation 사용
2. Our Navigation (양안 이미지 기반으로 네비게이션)
'''

import rospy

import cv2
import math
import time as t
import numpy as np
import matplotlib.pyplot as plt

from utils_Input import Image_load, Realsense_Image
from utils_Rotation import rotate_data          # function
from utils_Rectification import Rectification
from utils_SGBM import SGBM

from utils_Navigation import Navigation

from utils_SuperPixel import SuperPixelSampler
from utils_Link import Link_Adabins, Link_DenseDepth
from utils_Display import DISPLAY
from utils_Drone import Drone_CTRL

# Evaluation
from utils_Eval import evaluate


class WideStereo():
    # class declaration
    def __init__(self):
        # Option 
        self.Input_opt = 0      # Gazebo:0, Realsense:1
        self.MDE_opt = 0        # Adabins:0, DenseDepth:1
        
        # Load input data
        if self.Input_opt == 0:
            self.input = Image_load()
        if self.Input_opt == 1:
            self.input = Realsense_Image()
        
        # Monocular Depth Estimation
        if self.MDE_opt == 0:
            self.adabins = Link_Adabins()
        if self.MDE_opt == 1:
            self.densedepth = Link_DenseDepth()
        
        # Rectification
        self.rect = Rectification()
        


    # Input image(RGB & GT)
    def input_data(self):
        print("\n\n")
        print('Starting Input_Image computation...')
        st = t.time()
        self.left_RGB, self.right_RGB   = self.input.ROS_RGB()
        et = t.time()
        print("\tInput_Image execution time \t= {:.3f}s".format(et-st))
        
    
    
    # Monocular Depth Estimation 
    def MDE(self):
        print('Starting Monocular_Depth_Estimation computation...')
        st = t.time()
        left_mde_input  = self.left_RGB.copy()
        right_mde_input = self.right_RGB.copy()
        if self.MDE_opt==0:
            self.left_MDE   = self.adabins.predict(left_mde_input)
            self.right_MDE  = self.adabins.predict(right_mde_input)
        if self.MDE_opt==1:
            self.left_MDE   = self.densedepth.predict(left_mde_input)
            self.right_MDE  = self.densedepth.predict(right_mde_input)
            print(self.right_MDE.shape)
        et = t.time()
        print("\tMonocular_Depth_Estimation execution time \t= {:.3f}s".format(et-st))
        
        
    # Rotation as preprocessing of rectification
    def rotation(self):
        print('Starting Rotation computation...')
        st = t.time()
        self.rect_left_MDE, self.rect_right_MDE = rotate_data(self.left_MDE, self.right_MDE)
        et = t.time()
        print("\tRotation execution time \t\t\t= {:.3f}s".format(et-st))
        
        
    # Rectification
    def rectification(self):
        # preprocessing for rectification
        self.rotation()
        
        print('Starting Rectification computation...')
        st = t.time()
        self.rect_left_RGB, self.rect_right_RGB = self.rect.remap_img(self.left_RGB, self.right_RGB)
        self.rect_left_MDE, self.rect_right_MDE = self.rect.remap_img(self.rect_left_MDE, self.rect_right_MDE)
        et = t.time()
        print("\tRectification execution time \t\t\t= {:.3f}s".format(et-st))
        
            
    
if __name__ == '__main__':
    try:
        WS = WideStereo()
        
        # rate = rospy.Rate(1)
        
        while not rospy.is_shutdown():
            '''
            "Common Part"
            ''' 
            WS.input_data()
            WS.MDE()
            WS.rectification()
            
            
            
    except rospy.ROSInterruptException:
        pass
    finally:
        # WS.trajectory_save()
        print("\n\n// END //\n")
