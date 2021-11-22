#!/usr/bin/env python3

'''
1. For the 28th Samsung Human Tech
'''

import rospy

import time as t
import numpy as np

from utils_Input import Image_load
from utils_Rectification import Rectification

from utils_Link import Link_Adabins

'''
1. MDE's depth data scaling with StereoMatching 
2. Navigation with depth image
'''
class HumanTech():
    # class declaration
    def __init__(self):
        self.input = Image_load()
        self.rect = Rectification()
        self.adabins = Link_Adabins()

    # Input image(RGB & GT)
    def input_data(self):
        self.left_color, self.right_color     = self.input.ROS_RGB()
        self.left_depth, self.right_depth     = self.input.ROS_GT()

    # Rectification
    def rectification(self):
        print('Starting rectification computation...')
        st = t.time()
        self.rect_left_color, self.rect_right_color = self.rect.remap_img(self.left_color, self.right_color)
        self.rect_left_depth, self.rect_right_depth = self.rect.remap_img(self.left_depth, self.right_depth)
        et = t.time()
        print("\tRectification execution time = {:.5f}s".format(et-st))

    # Monocular Depth Estimation 
    # Adabins
    def MDE(self):
        # RGB와 Depth의 synchronization를 위해 토픽 사용 안함.
        print('Starting rectification computation...')
        st = t.time()
        self.adabins.
        et = t.time()
        print("\tRectification execution time = {:.5f}s".format(et-st))

    
if __name__ == '__main__':
    try:
        ht = HumanTech()
        
        
        r=rospy.Rate(10)    
        while not rospy.is_shutdown():
            ht.input_data()
            ht.rectification()
            
            
            r.sleep()

    except rospy.ROSInterruptException:
        pass
    finally:
        print("\n\n// END //\n")
