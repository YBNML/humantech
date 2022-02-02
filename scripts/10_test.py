#!/usr/bin/env python3

'''
< 00_our_Navi.py >
1. Monocular Depth Estimation 사용
2. Our Navigation (양안 이미지 기반으로 네비게이션)
'''

import rospy

import cv2
import time as t
import numpy as np
import matplotlib.pyplot as plt

from utils.input import Realsense_Image
from utils.link import Link_Adabins, Link_DenseDepth
from utils.rectification import Rectification
from utils.rotation import rotate_data
from utils.SGBM import SGBM
from utils.SuperPixel import SuperPixelSampler
from utils.navigation import Navigation


class IASL_KHJ():
    # class declaration
    def __init__(self):
        # option
        self.img_opt = 2    # 0:gazebo, 1:L515(1280x960), 2:l515(640x480)
        self.MDE_opt = 0    # 0:Adabins, 1:DenseDepth
        
        # Input image
        # self.input = Image_load()
        self.input = Realsense_Image()
        
        # Monocular Depth Estimation
        if self.MDE_opt==0:
            self.adabins = Link_Adabins()
        if self.MDE_opt == 1:
            self.densedepth = Link_DenseDepth()
        
        # Rectification
        self.rect = Rectification()
        
        # ZNCC
        self.sgbm = SGBM()
        
        # SuperPixel
        self.sp = SuperPixelSampler()
        
        # Navigation
        self.navi = Navigation()
        
        
    # Input image(RGB & GT)
    def input_data(self):
        print("\n\n")
        # self.left_RGB, self.right_RGB   = self.input.ROS_RGB()
        self.left_RGB, self.right_RGB   = self.input.test()
    
    
    # Monocular Depth Estimation 
    def MDE(self):
        st = t.time()
        left_mde_input  = self.left_RGB.copy()
        right_mde_input = self.right_RGB.copy()
        if self.MDE_opt==0:
            self.left_MDE   = self.adabins.predict(left_mde_input)
            self.right_MDE  = self.adabins.predict(right_mde_input)
        if self.MDE_opt==1:
            self.left_MDE   = self.densedepth.predict(left_mde_input)
            self.right_MDE  = self.densedepth.predict(right_mde_input)
        et = t.time()
        print("Monocular_Depth_Estimation : \t{:.3f}sec".format(et-st))
            
            
    # Rotation as preprocessing of rectification
    def rotation(self):
        st = t.time()
        self.rect_left_MDE, self.rect_right_MDE = rotate_data(self.left_MDE, self.right_MDE)
        et = t.time()
        print("Rotation : \t\t\t{:.3f}sec".format(et-st))
        
            
    # Rectification
    def rectification(self):
        # preprocessing for rectification
        self.rotation()
        
        st = t.time()
        self.rect_left_RGB, self.rect_right_RGB = self.rect.remap_img(self.left_RGB, self.right_RGB)
        self.rect_left_MDE, self.rect_right_MDE = self.rect.remap_img(self.rect_left_MDE, self.rect_right_MDE)
        et = t.time()
        print("Rectification : \t\t{:.3f}sec".format(et-st))
        
        
    # ZNCC & WTA - StereoMatching
    def stereomathcing(self):
        st = t.time()
        self.left_stereo_depth, self.right_stereo_depth =  self.sgbm.stereo_depth(self.rect_left_RGB, self.rect_right_RGB, self.rect_left_MDE, self.rect_right_MDE)
        self.crop_rect_left_RGB, self.crop_rect_right_RGB, self.crop_rect_left_MDE, self.crop_rect_right_MDE = self.sgbm.get_crop()
        # raw image size = 640x480
        # crop image size = 640x420
        et = t.time()
        print('ZNCC&WTA :\t\t\t{:.3f}sec'.format(et-st))
        
    
    # Scaling for scaling   
    def superpixel(self):
        st = t.time()
        __, self.left_scaling_factor = self.sp.superPixel(self.crop_rect_left_RGB, self.left_stereo_depth, self.crop_rect_left_MDE)
        __, self.right_scaling_factor = self.sp.superPixel(self.crop_rect_right_RGB, self.right_stereo_depth, self.crop_rect_right_MDE)
        et = t.time()
        print('SuperPixel :\t\t\t{:.3f}sec'.format(et-st))
    
    
    # Depth scaling
    def scaling(self):
        print('Starting Scaling computation...')
        st = t.time()
        if np.min(self.crop_rect_left_MDE[:,200:500])<0.8 or np.min(self.crop_rect_right_MDE[:,140:440])<0.8:
            self.left_scaling_factor=1
            self.right_scaling_factor=1
            print("@@@")
        print("\t", np.min(self.crop_rect_left_MDE[:,200:500]), np.min(self.crop_rect_right_MDE[:,140:440]))
            
            
        self.scaled_left_MDE = self.rect_left_MDE * self.left_scaling_factor
        self.scaled_right_MDE = self.rect_right_MDE * self.right_scaling_factor
        print("\tScaling Factor : " + str(self.left_scaling_factor) + "  " + str(self.right_scaling_factor))
        et = t.time()
        print('\tScaling execution time \t\t\t\t= {:.3f}s'.format(et-st))
        
    
    # Preprocessing for Navigation
    def navi_preprocessing(self):
        print('Starting Navi\'s Preprocessing computation...')
        st = t.time()
        # Crop
        self.merge_rect_left_RGB = self.rect_left_RGB[20:460,:320]
        self.merge_rect_right_RGB = self.rect_right_RGB[20:460,320:]
        self.merge_rect_left_MDE = self.scaled_left_MDE[20:460,:320]
        self.merge_rect_right_MDE = self.scaled_right_MDE[20:460,320:]
        self.merge_left_stereo_depth = self.left_stereo_depth[20:460,:320]
        self.merge_right_stereo_depth = self.right_stereo_depth[20:460,320:]
        # Superpixel
        self.left_seg_center, __ = self.sp.superPixel(self.merge_rect_left_RGB, self.merge_left_stereo_depth, self.merge_rect_left_MDE)
        self.right_seg_center, __ = self.sp.superPixel(self.merge_rect_right_RGB, self.merge_right_stereo_depth, self.merge_rect_right_MDE)
        self.right_seg_center[:,0] = self.right_seg_center[:,0] + 320
        et = t.time()
        print('\tNavi\'s Preprocessing execution time \t\t= {:.3f}s'.format(et-st))    
    
    
    # Drone Navigation in 3D-space
    def navigation(self):
        st = t.time()
        self.left_angular_velocity, self.left_thrust, self.left_forward_speed = self.navi.avoidObstacle2(self.left_seg_center)
        self.right_angular_velocity, self.right_thrust, self.right_forward_speed = self.navi.avoidObstacle2(self.right_seg_center)
        
        self.angular_velocity = self.left_angular_velocity + self.right_angular_velocity
        self.thrust = self.left_thrust + self.right_thrust
        
        if self.left_forward_speed >= self.right_forward_speed:
            self.forward_speed = self.right_forward_speed
        if self.left_forward_speed < self.right_forward_speed:
            self.forward_speed = self.left_forward_speed
        
        print(self.angular_velocity, self.thrust, self.forward_speed)
        
        et = t.time()
        print('Navigation :\t\t\t{:.3f}sec'.format(et-st))
        
        
    # Display
    def display(self):
        plt.subplot(5,2,1)
        plt.imshow(self.left_RGB)
        plt.subplot(5,2,2)
        plt.imshow(self.right_RGB)
        
        plt.subplot(5,2,3)
        plt.imshow(self.left_MDE)
        plt.subplot(5,2,4)
        plt.imshow(self.right_MDE)
        
        plt.subplot(5,2,5)
        plt.imshow(self.rect_left_RGB)
        plt.subplot(5,2,6)
        plt.imshow(self.rect_right_RGB)
        
        plt.subplot(5,2,7)
        plt.imshow(self.rect_left_MDE)
        plt.subplot(5,2,8)
        plt.imshow(self.rect_right_MDE)
        

        plt.subplot(5,2,9)
        plt.imshow(self.left_stereo_depth)
        plt.subplot(5,2,10)
        plt.imshow(self.right_stereo_depth)
        
        plt.show()
    
    
if __name__ == '__main__':
    try:
        iasl = IASL_KHJ()
        
        while not rospy.is_shutdown():
            '''
            "Common Part"
            ''' 
            iasl.input_data()
            iasl.MDE()
            iasl.rectification()
            
            '''
            "Depth Scaling Part"
            '''
            iasl.stereomathcing()
            iasl.superpixel()
            iasl.scaling()
            
            
            '''
            "Navigation Part"
            '''
            iasl.navi_preprocessing()
            iasl.navigation()
            
            
            
            '''
            "Display"
            '''
            iasl.display()
            
            
            
    except rospy.ROSInterruptException:
        pass
    finally:
        print("\n\n// END //\n")
