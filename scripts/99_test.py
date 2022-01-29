import cv2
import numpy as np

import matplotlib.pyplot as plt
from utils_Link import Link_Adabins

class MDE():
    def __init__(self):
        self.adabins = Link_Adabins()
        
        self.input = cv2.imread("img1.jpg")
        self.input = cv2.cvtColor(self.input, cv2.COLOR_BGR2RGB)
        self.input = self.input[64:784,120:1080,:]
        self.input = cv2.resize(self.input, (640, 480), interpolation=cv2.INTER_AREA)

        # self.input = cv2.imread("img2.JPG")
        # self.input = cv2.cvtColor(self.input, cv2.COLOR_BGR2RGB)
        # # self.input = self.input[66:466,78:618,:]
        # self.input = self.input[66:466,774:1314,:]
        # self.input = cv2.resize(self.input, (640, 480), interpolation=cv2.INTER_AREA)

        print(self.input)
        print(self.input.shape)
        
    def predict(self):
        self.depth = self.adabins.predict(self.input)
        
        
        plt.subplot(1,2,1)
        plt.imshow(self.input)
        plt.title("Color image")
        
        plt.subplot(1,2,2)
        plt.imshow(self.depth, cmap='magma_r')
        plt.title("Depth image(Adabins)")
        plt.colorbar()
        plt.show()
        

if __name__ == '__main__': 
    mde = MDE()
    
    mde.predict()