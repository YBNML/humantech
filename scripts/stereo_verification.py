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
from utils_Navi import avoidObstacle, display_navi

from utils_ZNCC import zncc_left, zncc_right
import utils_stereo_matching as sm

from utils_SuperPixel import SuperPixelSampler
from utils_Link import Link_Adabins
from utils_Display import DISPLAY
from utils_Drone import Drone_CTRL

from utils_SGBM import SGBM
'''
1. MDE's depth data scaling with StereoMatching 
2. Navigation with depth image
'''
class HumanTech():
    # class declaration
    def __init__(self):
        # Load init setting data
        self.param  = parameter()
        self.D = self.param.get_D()     # Max disparity
        self.R = self.param.get_R()     # Size of window to consider around the scan line point
        self.BL = self.param.get_BL()
        self.F = self.param.get_f()
        
        # Load input data from gazabo
        self.input = Image_load()
        # Adabins
        self.adabins = Link_Adabins()
        # Rectification
        self.rect = Rectification()
        
        # ZNCC
        self.sgbm = SGBM()
        
        # SuperPixel
        self.sp = SuperPixelSampler()
        
        # Drone Ctrl
        self.drone = Drone_CTRL()


    # Input image(RGB & GT)
    def input_data(self):
        print("\n\n")
        self.left_RGB, self.right_RGB   = self.input.ROS_RGB()
        self.left_GT, self.right_GT     = self.input.ROS_GT()
        # self.left_GT = self.left_GT.reshape((480,640))
        # self.right_GT = self.right_GT.reshape((480,640))
        
        # self.left_RGB, self.right_RGB   = self.input.test_RGB()
        # self.left_GT, self.right_GT     = self.input.test_GT()
    
        # self.left_GT = np.nan_to_num(self.left_GT, copy=True)
        # self.right_GT = np.nan_to_num(self.right_GT, copy=True)
        # print(self.left_GT)
    
    # Monocular Depth Estimation 
    def MDE(self):
        print('Starting Monocular_Depth_Estimation computation...')
        st = t.time()
        left_mde_input  = self.left_RGB
        right_mde_input = self.right_RGB
        self.left_MDE   = self.adabins.predict(left_mde_input)
        self.right_MDE  = self.adabins.predict(right_mde_input)
        et = t.time()
        print("\tMonocular_Depth_Estimation execution time \t= {:.3f}s".format(et-st))
        
        
    # Rotation as preprocessing of rectification
    def rotation(self):
        print('Starting Rotation computation...')
        st = t.time()
        self.rect_left_MDE, self.rect_right_MDE = rotate_data(self.left_MDE, self.right_MDE)
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
        self.rect_left_MDE, self.rect_right_MDE = self.rect.remap_img(self.rect_left_MDE, self.rect_right_MDE)
        et = t.time()
        print("\tRectification execution time \t\t\t= {:.3f}s".format(et-st))
        
        # GT
        # self.rect_left_GT, self.rect_right_GT = self.rect.remap_img(self.rect_left_GT, self.rect_right_GT)

    # Crop for StereoMatching's preprocessing
    def crop(self):
        # In 640x480 case
        self.crop_rect_left_RGB = self.rect_left_RGB[40:440,140:500]
        self.crop_rect_right_RGB = self.rect_right_RGB[40:440,140:500]
        self.crop_rect_left_MDE = self.rect_left_MDE[40:440,140:500]
        self.crop_rect_right_MDE = self.rect_right_MDE[40:440,140:500]
        
    # Crop for StereoMatching's preprocessing
    def blur(self):
        # Color image blur
        self.crop_rect_left_RGB      = cv2.GaussianBlur(self.crop_rect_left_RGB, (3,3), 0, 0)
        self.crop_rect_right_RGB     = cv2.GaussianBlur(self.crop_rect_right_RGB, (3,3), 0, 0)
        
    # Crop for StereoMatching's preprocessing
    def rgb2gray(self):
        # convert RGB to Gray for stereo matching
        self.gray_crop_rect_left_RGB     = rgb2gray(self.crop_rect_left_RGB)
        self.gray_crop_rect_right_RGB    = rgb2gray(self.crop_rect_right_RGB)
    
    # StereoMatching's preprocessing
    def preprocessing(self):
        print('Starting Preprocessing computation...')
        st = t.time()
        self.crop()
        self.blur()
        self.rgb2gray()
        et = t.time()
        print("\tPreprocessing execution time \t\t\t= {:.3f}s".format(et-st))
        
    # ZNCC & WTA - StereoMatching
    def stereomathcing(self):
        print('Starting ZNCC&WTA computation...')
        st = t.time()
        # Adaptive window size
        G_x = np.array([[-1,0,1]])
        l_edge = cv2.filter2D(self.gray_crop_rect_left_RGB, cv2.CV_64F, G_x)
        r_edge = cv2.filter2D(self.gray_crop_rect_right_RGB, cv2.CV_64F, G_x)
        l_edge = np.abs(l_edge)
        r_edge = np.abs(r_edge)
        l_edge_xsum = np.sum(l_edge,axis=1)
        r_edge_xsum = np.sum(r_edge,axis=1)
        # Disparity
        # self.gray_crop_rect_left_RGB = self.gray_crop_rect_left_RGB/255
        # self.gray_crop_rect_right_RGB = self.gray_crop_rect_right_RGB/255
        # print(t.time()-st)
        # self.left_disparity   = zncc_left(self.gray_crop_rect_left_RGB, self.gray_crop_rect_right_RGB, self.D, self.R, l_edge_xsum)
        self.left_stereo_depth, self.right_stereo_depth =  self.sgbm.stereo_depth(self.gray_crop_rect_left_RGB, self.gray_crop_rect_right_RGB)
        print(t.time()-st)
        # self.right_disparity  = zncc_right(self.gray_crop_rect_left_RGB, self.gray_crop_rect_right_RGB, self.D, self.R, r_edge_xsum)
        # print(t.time()-st)
        et = t.time()
        print('\tZNCC & WTA execution time \t\t\t= {:.3f}s'.format(et-st))
        
    
    def superpixel(self):
        print('Starting SuperPixel computation...')
        st = t.time()
        __, self.left_scaling_factor = self.sp.superPixel(self.crop_rect_left_RGB, self.left_stereo_depth, self.crop_rect_left_MDE)
        __, self.right_scaling_factor = self.sp.superPixel(self.crop_rect_right_RGB, self.right_stereo_depth, self.crop_rect_right_MDE)
        et = t.time()
        print('\tSuperPixel execution time \t\t\t= {:.3f}s'.format(et-st))
    
    # Depth scaling
    def scaling(self):
        print('Starting Scaling computation...')
        st = t.time()
        self.scaled_left_MDE = self.left_MDE * self.left_scaling_factor
        self.scaled_right_MDE = self.right_MDE * self.right_scaling_factor
        print("\tScaling Factor : " + str(self.left_scaling_factor) + "  " + str(self.right_scaling_factor))
        et = t.time()
        print('\tScaling execution time \t\t\t\t= {:.3f}s'.format(et-st))
  
    
if __name__ == '__main__':
    try:
        ht = HumanTech()
           
        while not rospy.is_shutdown():
            "Common Part"
            ht.input_data()
            ht.MDE()
            ht.rectification()
            
            "Depth Scaling Part"
            ht.preprocessing()
            ht.stereomathcing()
            ht.superpixel()
            ht.scaling()
            
            
    except rospy.ROSInterruptException:
        pass
    finally:
        print("\n\n// END //\n")
