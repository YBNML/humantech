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


class HumanTech():
    # class declaration
    def __init__(self):
        # Load input data from gazabo
        # self.input = Image_load()
        self.input = Realsense_Image()
        
        # Monocular Depth Estimation
        self.MDE_opt = 0     # Adabins:0, DenseDepth:1
        if self.MDE_opt==0:
            self.adabins = Link_Adabins()
        if self.MDE_opt == 1:
            self.densedepth = Link_DenseDepth()
        


    # Input image(RGB & GT)
    def input_data(self):
        print("\n\n")
        self.left_RGB, self.right_RGB   = self.input.RS_RGB()
    
    
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
        
    
    def display(self):
        # print(self.left_RGB)
        cv2.imshow("Left_RGB",self.left_RGB)
        cv2.imshow("Right_RGB",self.right_RGB)
        
        __, lviz = DISPLAY(self.left_MDE)
        __, rviz = DISPLAY(self.right_MDE)
        
        cv2.imshow("Left_MDE",lviz)
        cv2.imshow("Right_MDE",rviz)
        
        cv2.waitKey(10)
        

    
if __name__ == '__main__':
    try:
        ht = HumanTech()
        
        # rate = rospy.Rate(1)
        
        while not rospy.is_shutdown():
            '''
            "Common Part"
            ''' 
            ht.input_data()
            ht.MDE()
            
            '''
            "Display"
            '''
            ht.display()
            
            
            
    except rospy.ROSInterruptException:
        pass
    finally:
        # ht.trajectory_save()
        print("\n\n// END //\n")
