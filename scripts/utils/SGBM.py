#!/usr/bin/env python3

import cv2
import time as t
import numpy as np
# from skimage.color import rgb2gray
import matplotlib.pyplot as plt

from utils_Parameter import parameter

class SGBM:
    def __init__(self):
        # Load init setting data
        self.param  = parameter()
        self.D      = self.param.get_D()     # Max disparity
        self.R      = self.param.get_R()     # Size of window to consider around the scan line point
        self.BL     = self.param.get_BL()
        self.F      = self.param.get_f()
        
        self.min_disparity = 1
        

    def stereo_depth(self, left_img, right_img, left_mde, right_mde):
        self.left_img = left_img
        self.right_img = right_img
        self.left_mde = left_mde
        self.right_mde = right_mde
        
        self.preprocessing()
        
        self.disparity()
        
        self.right_disp = -self.right_disp
        
        self.left_disp = np.clip(self.left_disp, 0.01, 200)
        self.right_disp = np.clip(self.right_disp, 0.01, 200)
        
        left_stereo     = 0.2 * self.F /self.left_disp
        right_sterep    = 0.2 * self.F /self.right_disp
        
        return left_stereo, right_sterep
    
    
    def preprocessing(self):
        self.crop()
        self.blur()
        self.rgb2gray()
        
        # Adaptive window size
        G_x = np.array([[-1,0,1]])
        l_edge = cv2.filter2D(self.gray_crop_left_RGB, cv2.CV_64F, G_x)
        r_edge = cv2.filter2D(self.gray_crop_right_RGB, cv2.CV_64F, G_x)
        l_edge = np.abs(l_edge)
        r_edge = np.abs(r_edge)
        self.l_edge_sum = np.sum(l_edge)
        self.r_edge_sum = np.sum(r_edge)
        
        self.tempR = 5
        if (self.l_edge_sum + self.r_edge_sum) > 10000:
            self.tempR = 4
        elif (self.l_edge_sum + self.r_edge_sum) < 3000:
            self.tempR = 4
            
            
    # Crop for StereoMatching's preprocessing
    def crop(self):
        # In 640x480 case
        if self.left_img.shape[0]==480 and self.left_img.shape[1]==640:
            # self.numDisparities = 140
            # self.crop_left_img      = self.left_img[40:440,:]
            # self.crop_right_img     = self.right_img[40:440,:]
            # self.crop_left_mde      = self.left_mde[40:440,:]
            # self.crop_right_mde     = self.right_mde[40:440,:]
            self.numDisparities = 180
            self.crop_left_img      = self.left_img[30:450,:]
            self.crop_right_img     = self.right_img[30:450,:]
            self.crop_left_mde      = self.left_mde[30:450,:]
            self.crop_right_mde     = self.right_mde[30:450,:]
            
        # In 1280x720 case
        if self.left_img.shape[0]==720 and self.left_img.shape[1]==1280:
            self.numDisparities = 440
            self.crop_left_img      = self.left_img[35:685,:]
            self.crop_right_img     =self.right_img[35:685,:]
            self.crop_left_mde      = self.left_mde[35:685,:]
            self.crop_right_mde     =self.right_mde[35:685,:]
        
    # Blur for StereoMatching's preprocessing
    def blur(self):
        # Color image blur
        self.crop_left_img      = cv2.GaussianBlur(self.crop_left_img, (3,3), 0, 0)
        self.crop_right_img     = cv2.GaussianBlur(self.crop_right_img, (3,3), 0, 0)
        
    # GRAY for StereoMatching's preprocessing
    def rgb2gray(self):
        # convert RGB to Gray for stereo matching
        self.gray_crop_left_RGB     = cv2.cvtColor(self.crop_left_img, cv2.COLOR_RGB2GRAY)
        self.gray_crop_right_RGB    = cv2.cvtColor(self.crop_right_img, cv2.COLOR_RGB2GRAY)


    
    def disparity(self):
        
        # https://docs.opencv.org/trunk/d2/d85/classcv_1_1StereoSGBM.html
        self.left_matcher    = cv2.StereoSGBM_create(minDisparity = self.min_disparity,
                                                numDisparities = self.numDisparities,
                                                blockSize = self.tempR,
                                                P1 = 8 * 3 * self.tempR**2,
                                                P2 = 32 * 3 * self.tempR**2,
                                                disp12MaxDiff = 1,
                                                uniquenessRatio = 10,
                                                speckleWindowSize = 100,
                                                speckleRange = 32)
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
        
        self.gray_crop_left_RGB     = self.gray_crop_left_RGB.astype(np.uint8)
        self.gray_crop_right_RGB    = self.gray_crop_right_RGB.astype(np.uint8)
        
        self.left_disp   = self.left_matcher.compute(self.gray_crop_left_RGB, self.gray_crop_right_RGB)
        self.right_disp  = self.right_matcher.compute(self.gray_crop_right_RGB, self.gray_crop_left_RGB)

        self.left_disp = self.left_disp.astype(np.float32) / 16
        self.right_disp = self.right_disp.astype(np.float32) / 16
        
    def get_crop(self):
        return self.crop_left_img, self.crop_right_img, self.crop_left_mde, self.crop_right_mde 
        