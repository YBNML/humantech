#!/usr/bin/env python3

import cv2
import numpy as np
from skimage.color import rgb2gray

from utils_Parameter import parameter

class SGBM:
    def __init__(self):
        # Load init setting data
        self.param  = parameter()
        self.D      = self.param.get_D()     # Max disparity
        self.R      = self.param.get_R()     # Size of window to consider around the scan line point
        self.BL     = self.param.get_BL()
        self.F      = self.param.get_f()
        
        min_disparity = 0
        max_disparity = self.D
        
        # https://docs.opencv.org/trunk/d2/d85/classcv_1_1StereoSGBM.html
        self.left_matcher    = cv2.StereoSGBM_create(minDisparity = min_disparity,
                                                numDisparities = max_disparity,
                                                blockSize = self.R,
                                                P1 = 8 * 3 * self.R**2,
                                                P2 = 32 * 3 * self.R**2,
                                                disp12MaxDiff = 1,
                                                uniquenessRatio = 10,
                                                speckleWindowSize = 100,
                                                speckleRange = 32)
        
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)


    def stereo_depth(self):
        print("1@@")
        self.left_img = left_img1
        self.right_img = right_img2
        self.left_mde = left_mde3
        self.right_mde = right_mde4
        
        # self.preprocessing()
        
        # self.disparity()
        
        # left_stereo = self.BL*self.F/self.left_disp
        # right_sterep = self.BL*self.F/self.right_disp
        
        return 0, 0

    # Crop for StereoMatching's preprocessing
    def crop(self):
        # In 640x480 case
        self.crop_left_img      = self.left_img[40:440,140:500]
        self.crop_right_img     = self.right_img[40:440,140:500]
        self.crop_left_mde      = self.left_mde[40:440,140:500]
        self.crop_right_mde     = self.right_mde[40:440,140:500]
        
    # Blur for StereoMatching's preprocessing
    def blur(self):
        # Color image blur
        self.crop_left_img      = cv2.GaussianBlur(self.crop_left_img, (3,3), 0, 0)
        self.crop_right_img     = cv2.GaussianBlur(self.crop_right_img, (3,3), 0, 0)
        
    # GRAY for StereoMatching's preprocessing
    def rgb2gray(self):
        # convert RGB to Gray for stereo matching
        self.gray_crop_left_RGB     = rgb2gray(self.crop_left_img)
        self.gray_crop_right_RGB    = rgb2gray(self.crop_right_img)

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
        self.l_edge_xsum = np.sum(l_edge,axis=1)
        self.r_edge_xsum = np.sum(r_edge,axis=1)
        
    

    def disparity(self):
        self.gray_crop_left_RGB     = self.gray_crop_left_RGB.astype(np.uint8)
        self.gray_crop_right_RGB    = self.gray_crop_right_RGB.astype(np.uint8)
        self.left_disp   = self.left_matcher.compute(self.gray_crop_left_RGB, self.gray_crop_right_RGB)
        self.right_disp  = self.right_matcher.compute(self.gray_crop_right_RGB, self.gray_crop_left_RGB)

        # normalising disparities for saving/display
        # disparity_norm = disparity.astype(np.float32) / 16
        # left_disp_norm = left_disp.astype(np.float32) / 16

        
    def get_crop(self):
        return self.crop_left_img, self.crop_right_img, self.crop_left_mde, self.crop_right_mde 
        