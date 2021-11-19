#!/usr/bin/env python3

import os
import cv2
import sys
import math
import time as t
import numpy as np

from pathlib import Path


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]                  # Current Directory

class Image_load():
    '''
    Received image from dataset.
    '''
    def __init__(self):
        self.count=0

    def load_rgb(self):
        '''
        Load RGB image
        '''
        DIR = str(ROOT) + "/dataset/rgb/"
        left_color = cv2.imread(DIR+"left_color"+str(self.count)+".npy")
        right_color = cv2.imread(DIR+"right_color"+str(self.count)+".npy")
        return left_rgb, right_rgb

    def load_gt(self):
        '''
        Load Ground Truth
        '''
        DIR = str(ROOT) + "/dataset/gt/"
        left_gt = cv2.imread(DIR+"left_depth"+str(self.count)+".npy")
        right_gt = cv2.imread(DIR+"right_depth"+str(self.count)+".npy")
        return left_gt, right_gt
    
    def load_mde(self):
        '''
        Load Monocular_Depth_Estimation(MDE)
        '''
        mde = "adabins"     # choose MDE model (adabins, bts, densedepth)
        DIR = str(ROOT) + "/dataset/" + mde +"/"
        left_mde = cv2.imread(DIR + "left_" + mde + str(self.count) + ".npy")
        right_mde = cv2.imread(DIR + "right_" + mde + str(self.count) + ".npy")
        return left_mde, right_mde