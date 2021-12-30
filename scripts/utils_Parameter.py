#!/usr/bin/env python3

class parameter:
    def __init__(self):
        
        
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

        # accX tolerance 
        self.X = 3
        
        # Base Line
        self.BL = 0.2

        # number of example in the example list
        self.example_number = 1


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

    def get_example_number(self):
        return self.example_number

    def get_D(self):
        return self.D

    def get_R(self):
        return self.R

    def get_X(self):
        return self.X
    
    def get_BL(self):
        return self.BL