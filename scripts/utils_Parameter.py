#!/usr/bin/env python3

class parameter:
    def __init__(self):
        # gazebo = 0
        # realsense L515 = 1
        # realsense D435i = 2
        opt = 0
        
        # Gazebo's rotors
        if opt == 0:
            # camera parameter
            self.px = 320.5
            self.py = 240.5
            self.fx = 361.69539859647745
            self.fy = 361.69539859647745
            # Image Size
            self.img_width = 640
            self.img_height = 480
            # Max disparity
            self.D = 128
            # Size of window to consider around the scan line point
            self.R = 3
            # Base Line
            self.BL = 0.2       # 20cm = 0.2m
            
        # Realsense L515
        if opt == 1:
            # camera parameter
            self.px = 328.741
            self.py = 235.728
            self.fx = 601.439
            self.fy = 601.622
            # Image Size
            self.img_width = 640
            self.img_height = 480
            # Max disparity
            self.D = 128
            # Size of window to consider around the scan line point
            self.R = 3
            # Base Line
            self.BL = 0.2       # 20cm = 0.2m


    def get_px(self):
        return self.px
    def get_py(self):
        return self.py

    def get_f(self):
        return (self.fx+self.fy)/2
    def get_fx(self):
        return self.fx
    def get_fy(self):
        return self.fy

    def get_img_w(self):
        return self.img_width
    def get_img_h(self):
        return self.img_height

    def get_D(self):
        return self.D

    def get_R(self):
        return self.R

    def get_X(self):
        return self.X
    
    def get_BL(self):
        return self.BL