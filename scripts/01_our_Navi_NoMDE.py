#!/usr/bin/env python3

'''
< 00_our_Navi.py >
1. Monocular Depth Estimation 사용 안함
2. Our Navigation (양안 이미지 기반으로 네비게이션)
'''

import rospy

import cv2
import math
import time as t
import numpy as np
import matplotlib.pyplot as plt

from utils_Input import Image_load
from utils_Rotation import rotate_data          # function
from utils_Rectification import Rectification
from utils_Navigation import Navigation

from utils_SuperPixel import SuperPixelSampler
from utils_Display import DISPLAY
from utils_Drone import Drone_CTRL


class HumanTech():
    # class declaration
    def __init__(self):
        # Load input data from gazabo
        self.input = Image_load()
        
        # Rectification
        self.rect = Rectification()
        
        # SuperPixel
        self.sp = SuperPixelSampler()
        
        # Drone Ctrl
        self.drone = Drone_CTRL()
        
        self.previous_yaw = 0
        
        self.navi = Navigation()


    # Input image(RGB & GT)
    def input_data(self):
        print("\n\n")
        self.left_RGB, self.right_RGB   = self.input.ROS_RGB()
        self.left_GT, self.right_GT     = self.input.ROS_GT()
        
        self.left_GT    = self.left_GT.reshape((480,640))
        self.right_GT   = self.right_GT.reshape((480,640))
        
        
    # Rotation as preprocessing of rectification
    def rotation(self):
        print('Starting Rotation computation...')
        st = t.time()
        self.left_GT = np.array(self.left_GT, dtype=np.float64)
        self.right_GT = np.array(self.right_GT, dtype=np.float64)
        self.rect_left_GT, self.rect_right_GT = rotate_data(self.left_GT, self.right_GT)
        et = t.time()
        print("\tRotation execution time \t\t\t= {:.3f}s".format(et-st))
        
        
    # Rectification
    def rectification(self):
        # preprocessing for rectification
        self.rotation()
        
        print('Starting Rectification computation...')
        st = t.time()
        self.rect_left_RGB, self.rect_right_RGB = self.rect.remap_img(self.left_RGB, self.right_RGB)
        self.rect_left_GT, self.rect_right_GT   = self.rect.remap_img(self.rect_left_GT, self.rect_right_GT)
        et = t.time()
        print("\tRectification execution time \t\t\t= {:.3f}s".format(et-st))
        

    
    def navi_preprocessing(self):
        print('Starting Navi\'s Preprocessing computation...')
        st = t.time()
        # Crop
        self.merge_rect_left_RGB = self.rect_left_RGB[105:375,:320]
        self.merge_rect_right_RGB = self.rect_right_RGB[105:375,320:]
        self.merge_rect_left_GT = self.rect_left_GT[105:375,:320]
        self.merge_rect_right_GT = self.rect_right_GT[105:375,320:]
        # Superpixel
        self.left_seg_center, __ = self.sp.superPixel(self.merge_rect_left_RGB, self.merge_rect_left_GT, self.merge_rect_left_GT)
        self.right_seg_center, __ = self.sp.superPixel(self.merge_rect_right_RGB, self.merge_rect_right_GT, self.merge_rect_right_GT)
        self.right_seg_center[:,0] = self.right_seg_center[:,0] + 320
        et = t.time()
        print('\tNavi\'s Preprocessing execution time \t\t= {:.3f}s'.format(et-st))
        
    # Drone Navigation in 3D-space
    def navigation(self):
        print('Starting Navigation computation...')
        st = t.time()
        self.left_angular_velocity, self.left_thrust, self.left_forward_speed = self.navi.avoidObstacle2(self.left_seg_center)
        self.right_angular_velocity, self.right_thrust,  self.right_forward_speed = self.navi.avoidObstacle2(self.right_seg_center)
        
        self.angular_velocity = self.left_angular_velocity + self.right_angular_velocity
        self.thrust = self.left_thrust + self.right_thrust
        
        if self.left_forward_speed >= self.right_forward_speed:
            self.forward_speed = self.right_forward_speed
        if self.left_forward_speed < self.right_forward_speed:
            self.forward_speed = self.left_forward_speed
        
        if math.isnan(self.angular_velocity) == True:
            self.angular_velocity=0
            self.thrust = 0
        et = t.time()
        # print(self.yaw)
        print('\tNavigation execution time \t\t\t= {:.3f}s'.format(et-st))
    
    
    
    def drone_ctrl(self):
        print('Starting Drone_Control computation...')
        st = t.time()
        self.drone.update_ours(self.angular_velocity, self.thrust, self.forward_speed)
        et = t.time()
        print('\tDrone_Command execution time \t\t\t= {:.3f}s'.format(et-st))
        
    def trajectory(self):
        print('Starting Trajectory_save computation...')
        st = t.time()
        self.drone.trajectory()
        et = t.time()
        print('\tTrajectory_save execution time \t\t\t= {:.3f}s'.format(et-st))
      
    def trajectory_save(self):
        print('Starting Trajectory_save computation...')
        st = t.time()
        self.drone.trajectory_save()
        et = t.time()
        print('\tTrajectory_save execution time \t\t\t= {:.3f}s'.format(et-st))
        
    def drone_display(self):
        # self.merge_rect_left_GT_viz = DISPLAY(self.merge_rect_left_GT)
        # self.merge_rect_right_GT_viz = DISPLAY(self.merge_rect_right_GT)
        base_rgb = np.hstack((self.merge_rect_left_RGB,self.merge_rect_right_RGB))
        base_rgb = cv2.cvtColor(base_rgb, cv2.COLOR_BGR2RGB)
        
        cv2.imshow('Navigation',base_rgb)
        cv2.waitKey(1)
        # print()
        
    
if __name__ == '__main__':
    ht = HumanTech()
    try:   
        while not rospy.is_shutdown():
            # "Common Part"
            ht.input_data()
            ht.rectification()
            
            # "Navigation Part"
            ht.navi_preprocessing()
            ht.navigation()
            
            # Gazebo drone control Part
            ht.drone_ctrl()
            
            ht.trajectory()
            ht.drone_display()
            
            
    except rospy.ROSInterruptException:
        pass
    finally:
        ht.trajectory_save()
        print("\n\n// END //\n")
