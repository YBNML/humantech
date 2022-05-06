#!/usr/bin/env python3

'''
< 02_DroNet.py >
1. Monocular Depth Estimation 사용 안함
2. DroNet(단안 이미지 기반으로 네비게이션)함
'''

import rospy

import cv2
import time as t
import numpy as np
import matplotlib.pyplot as plt

from utils_Input import Image_load

from utils_Link import Link_DroNet
from utils_Display import DISPLAY
from utils_Drone import Drone_CTRL


class HumanTech():
    # class declaration
    def __init__(self):
        # Load input data from gazabo
        self.input = Image_load()
        
        # Navigation
        self.dronet = Link_DroNet()
        
        # Drone Ctrl
        self.drone = Drone_CTRL()
        


    # Input image(RGB & GT)
    def input_data(self):
        print("\n\n")
        self.center_RGB = self.input.ROS_RGB2()
        # self.center_GT  = self.input.ROS_GT2()
        

    # Drone Navigation in 3D-space
    def navigation(self):
        print('Starting Navigation computation...')
        st = t.time()
        
        # DroNet
        self.dronet.navigation(self.center_RGB)
        self.velocity = self.dronet.get_vel()
        self.steering = -self.dronet.get_steer()
        self.velocity = 0.5*(1-self.velocity)
        print(self.velocity, self.steering)
        self.steering = np.clip(self.steering, -0.25,+0.25)
        
        et = t.time()
        # print(self.yaw)
        print('\tNavigation execution time \t\t\t= {:.3f}s'.format(et-st))
    
    def drone_ctrl(self):
        print('Starting Drone_Control computation...')
        st = t.time()
        self.drone.update_dronet(self.velocity, self.steering)
        et = t.time()
        self.steering =self.steering
        print('\tDrone_Command execution time \t\t\t= {:.3f}s'.format(et-st))
        
    def drone_display(self):
        cv2.imshow('Navigation',self.center_RGB)
        cv2.waitKey(1)
        # print()
        
    
if __name__ == '__main__':
    try:
        ht = HumanTech()
           
        while not rospy.is_shutdown():
            # "Common Part"
            ht.input_data()
            
            # "Navigation Part"
            ht.navigation()
            
            # Gazebo drone control Part
            ht.drone_ctrl()
            
            ht.drone_display()
            
            
    except rospy.ROSInterruptException:
        pass
    finally:
        print("\n\n// END //\n")
