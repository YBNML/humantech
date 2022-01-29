#!/usr/bin/env python3


import cv2
# import numpy as np

# import matplotlib.pyplot as plt

from utils_Input import Realsense_Image

class MDE():
    def __init__(self):
        self.input = Realsense_Image()
        
    def input_data(self):
        
        while True:
            self.left_RGB, self.right_RGB  = self.input.RS_RGB()
        
            print(self.left_RGB.shape)
            cv2.imshow("left", self.left_RGB)
            cv2.imshow("right", self.right_RGB)
            cv2.waitKey(1)
            k = cv2.waitKey(33)
            if k==27:    # Esc key to stop
                break

if __name__ == '__main__': 
    mde = MDE()
    mde.input_data()