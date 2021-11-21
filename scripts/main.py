#!/usr/bin/env python3

'''
1. For the 28th Samsung Human Tech
'''

import rospy

import numpy as np

from utils_Input import Image_load

'''
1. MDE's depth data scaling with StereoMatching 
2. Navigation with depth image
'''
class HumanTech():
    def __init__(self):
        # class declaration
        self.input = Image_load()


    def input_data(self):
        # Input image(RGB & Depth)
        self.left_color, self.right_color     = self.input.ROS_RGB()
        self.left_depth, self.right_depth     = self.input.ROS_GT()

if __name__ == '__main__':
    try:
        ht = HumanTech()
        while True:
            ht.input_data()

        print("")
    finally:
        print("\n\n// END //\n")
