#!/usr/bin/env python3

'''
For the 28th Samsung Human Tech
'''

import rospy

import cv2
import time as t
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

from utils_Parameter import parameter
from utils_Input import Image_load
from utils_Rotation import rotate_data          # function
from utils_Rectification import Rectification
from utils_Navi import Navigation, avoidObstacle

from utils_ZNCC import zncc_left, zncc_right
import utils_stereo_matching as sm

from utils_SuperPixel import SuperPixelSampler
from utils_Link import Link_Adabins
from utils_Display import DISPLAY
from utils_Drone import Drone_CTRL

'''
장애물 회피(네비게이션)의 알고리즘 성능 확인을 위한 코드
main.py 알고리즘 구조를 그대로 사용한 것
'''
class HumanTech():
    # class declaration
    def __init__(self):
        # Load input data from gazabo
        self.input = Image_load()
        # Adabins
        self.adabins = Link_Adabins()
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
        
        # GT
        # self.rect_left_GT, self.rect_right_GT = rotate_data(self.left_GT, self.right_GT)
        
        
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
        self.left_seg_center, __ = self.sp.superPixel(self.merge_rect_left_RGB, self.rect_left_GT, self.rect_left_GT)
        self.right_seg_center, __ = self.sp.superPixel(self.merge_rect_right_RGB, self.rect_right_GT, self.rect_right_GT)
        self.right_seg_center[:,0] = self.right_seg_center[:,0] + 320
        # Merge 2 seg_center
        self.seg_center = np.concatenate((self.left_seg_center,self.right_seg_center), axis=0)
        et = t.time()
        print('\tNavi\'s Preprocessing execution time \t\t= {:.3f}s'.format(et-st))
        
        # view = np.hstack((self.merge_rect_left_GT, self.merge_rect_right_GT))
        # plt.imshow(view)
        # plt.show()
    
    # Drone Navigation in 3D-space
    def navigation(self):
        print('Starting Navigation computation...')
        st = t.time()
        view = np.hstack((self.merge_rect_left_GT, self.merge_rect_right_GT))
        self.yaw = self.navi.avoidObstacle(view)
        self.thrust = 0
        # self.yaw, self.thrust = avoidObstacle(self.seg_center)
        et = t.time()
        # print(self.yaw)
        print('\tNavigation execution time \t\t\t= {:.3f}s'.format(et-st))
    
    def drone_ctrl(self):
        print('Starting Drone_Control computation...')
        st = t.time()
        self.drone.update_desired(self.yaw, self.thrust)
        et = t.time()
        print('\tDrone_Command execution time \t\t\t= {:.3f}s'.format(et-st))
        
    def drone_display(self):
        # self.merge_rect_left_GT_viz = DISPLAY(self.merge_rect_left_GT)
        # self.merge_rect_right_GT_viz = DISPLAY(self.merge_rect_right_GT)
        base_rgb = np.hstack((self.merge_rect_left_RGB,self.merge_rect_right_RGB))
        base_depth = np.hstack((self.merge_rect_left_GT,self.merge_rect_right_GT))
        
        # plt.subplot(2,1,1)
        # plt.imshow(base_rgb)
        # plt.subplot(2,1,2)
        # plt.imshow(base_depth)
        # plt.show()
        
        # display_navi(base_img,10,10,10)
        cv2.imshow('Navigation',base_rgb)
        cv2.waitKey(1)
        # print()
        
    
if __name__ == '__main__':
    try:
        ht = HumanTech()
           
        while not rospy.is_shutdown():
            # "Common Part"
            ht.input_data()
            ht.rectification()
            
            # "Navigation Part"
            ht.navi_preprocessing()
            ht.navigation()
            
            # Gazebo drone control Part
            ht.drone_ctrl()
            
            # ht.drone_display()
            
            
    except rospy.ROSInterruptException:
        pass
    finally:
        print("\n\n// END //\n")
