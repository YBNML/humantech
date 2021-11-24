#!/usr/bin/env python3

'''
For the 28th Samsung Human Tech
'''

import rospy

import cv2
import time as t
import numpy as np

from utils_Input import Image_load
from utils_Rotation import rotate_data          # function
from utils_Rectification import Rectification


from utils_Link import Link_Adabins


from utils_Display import DISPLAY

'''
1. MDE's depth data scaling with StereoMatching 
2. Navigation with depth image
'''
class HumanTech():
    # class declaration
    def __init__(self):
        # Load input data from gazabo
        self.input = Image_load()
        # Adabins
        self.adabins = Link_Adabins()
        # Rotation 
        # self.rot = Rotation()
        # Rectification
        self.rect = Rectification()

    # Input image(RGB & GT)
    def input_data(self):
        print("\n\n")
        self.left_RGB, self.right_RGB   = self.input.ROS_RGB()
        self.left_GT, self.right_GT     = self.input.ROS_GT()

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
        
    # Rectification
    def rectification(self):
        # preprocessing
        self.rotation()
        
        print('Starting Rectification computation...')
        st = t.time()
        self.rect_left_RGB, self.rect_right_RGB = self.rect.remap_img(self.left_RGB, self.right_RGB)
        self.rect_left_MDE, self.rect_right_MDE = self.rect.remap_img(self.rect_left_MDE, self.rect_right_MDE)
        et = t.time()
        print("\tRectification execution time \t\t\t= {:.3f}s".format(et-st))

    # Crop for StereoMatching's preprocessing
    def crop(self):
        # In 640x480 case
        self.rect_left_RGB = self.rect_left_RGB[40:440,140:500]
        self.rect_right_RGB = self.rect_right_RGB[40:440,140:500]
        self.rect_left_MDE = self.rect_left_MDE[40:440,140:500]
        self.rect_right_MDE = self.rect_right_MDE[40:440,140:500]
    # Crop for StereoMatching's preprocessing
    def blur(self):
        # Color image blur
        self.rect_left_RGB      = cv2.GaussianBlur(self.rect_left_RGB, (3,3), 0, 0)
        self.rect_right_RGB     = cv2.GaussianBlur(self.rect_right_RGB, (3,3), 0, 0)
    
    # StereoMatching's preprocessing
    def preprocessing(self):
        print('Starting Preprocessing computation...')
        st = t.time()
        self.crop()
        self.blur()
        et = t.time()
        print("\tPreprocessing execution time \t\t\t= {:.3f}s".format(et-st))
        
    
    def display(self):
        RGB_viz = np.hstack((self.rect_left_RGB,self.rect_right_RGB))
        cv2.imshow("1. RGB_viz", RGB_viz)
        
        __, left_depth_viz = DISPLAY(self.left_GT)
        __, right_depth_viz = DISPLAY(self.right_GT)
        Depth_viz = np.hstack((left_depth_viz,right_depth_viz))
        cv2.imshow("1. Depth_viz", Depth_viz)
        
        
        __, left_mde_viz = DISPLAY(self.left_MDE)
        __, right_mde_viz = DISPLAY(self.right_MDE)
        mde_viz = np.hstack((left_mde_viz,right_mde_viz))
        cv2.imshow("1. MDE_viz", mde_viz)
        
        cv2.waitKey(10)

        # DISPLAY()
    
if __name__ == '__main__':
    try:
        ht = HumanTech()
        
        
        r=rospy.Rate(10)    
        while not rospy.is_shutdown():
            ht.input_data()
            ht.MDE()
            ht.rectification()
            ht.preprocessing()
            
            
            
            ht.display()
            
            r.sleep()

    except rospy.ROSInterruptException:
        pass
    finally:
        print("\n\n// END //\n")
