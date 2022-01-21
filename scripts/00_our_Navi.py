#!/usr/bin/env python3

'''
< 00_our_Navi.py >
1. Monocular Depth Estimation 사용
2. Our Navigation (양안 이미지 기반으로 네비게이션)
'''

import rospy

import cv2
import time as t
import numpy as np
import matplotlib.pyplot as plt

from utils_Input import Image_load
from utils_Rotation import rotate_data          # function
from utils_Rectification import Rectification
from utils_SGBM import SGBM

from utils_Navigation import Navigation

from utils_SuperPixel import SuperPixelSampler
from utils_Link import Link_Adabins, Link_DenseDepth
from utils_Display import DISPLAY
from utils_Drone import Drone_CTRL

# Evaluation
from utils_Eval import evaluate


class HumanTech():
    # class declaration
    def __init__(self):
        # Load input data from gazabo
        self.input = Image_load()
        
        # Monocular Depth Estimation
        self.MDE_opt = 1     # Adabins:0, DenseDepth:1
        if self.MDE_opt==0:
            self.adabins = Link_Adabins()
        if self.MDE_opt == 1:
            self.densedepth = Link_DenseDepth()
        
        # Rectification
        self.rect = Rectification()
        
        # ZNCC
        self.sgbm = SGBM()
        
        # SuperPixel
        self.sp = SuperPixelSampler()
        
        # Drone Ctrl
        self.drone = Drone_CTRL()
        
        self.navi = Navigation()


    # Input image(RGB & GT)
    def input_data(self):
        print("\n\n")
        self.left_RGB, self.right_RGB   = self.input.ROS_RGB()
        self.left_GT, self.right_GT     = self.input.ROS_GT()
        
        # self.left_RGB, self.right_RGB   = self.input.test_RGB()
        # self.left_GT, self.right_GT     = self.input.test_GT()
    
    
    # Monocular Depth Estimation 
    def MDE(self):
        print('Starting Monocular_Depth_Estimation computation...')
        st = t.time()
        left_mde_input  = self.left_RGB.copy()
        right_mde_input = self.right_RGB.copy()
        if self.MDE_opt==0:
            self.left_MDE   = self.adabins.predict(left_mde_input)
            self.right_MDE  = self.adabins.predict(right_mde_input)
        if self.MDE_opt==1:
            self.left_MDE   = self.densedepth.predict(left_mde_input)
            self.right_MDE  = self.densedepth.predict(right_mde_input)
            print(self.right_MDE.shape)
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
        # preprocessing for rectification
        self.rotation()
        
        print('Starting Rectification computation...')
        st = t.time()
        self.rect_left_RGB, self.rect_right_RGB = self.rect.remap_img(self.left_RGB, self.right_RGB)
        self.rect_left_MDE, self.rect_right_MDE = self.rect.remap_img(self.rect_left_MDE, self.rect_right_MDE)
        et = t.time()
        print("\tRectification execution time \t\t\t= {:.3f}s".format(et-st))
        
        
    # ZNCC & WTA - StereoMatching
    def stereomathcing(self):
        print('Starting ZNCC&WTA computation...')
        st = t.time()
        self.left_stereo_depth, self.right_stereo_depth =  self.sgbm.stereo_depth(self.rect_left_RGB, self.rect_right_RGB, self.rect_left_MDE, self.rect_right_MDE)
        self.crop_rect_left_RGB, self.crop_rect_right_RGB, self.crop_rect_left_MDE, self.crop_rect_right_MDE = self.sgbm.get_crop()
        et = t.time()
        print('\tZNCC & WTA execution time \t\t\t= {:.3f}s'.format(et-st))
        
    
    def superpixel(self):
        print('Starting SuperPixel computation...')
        st = t.time()
        __, self.left_scaling_factor = self.sp.superPixel(self.crop_rect_left_RGB, self.left_stereo_depth, self.crop_rect_left_MDE)
        __, self.right_scaling_factor = self.sp.superPixel(self.crop_rect_right_RGB, self.right_stereo_depth, self.crop_rect_right_MDE)
        et = t.time()
        print('\tSuperPixel execution time \t\t\t= {:.3f}s'.format(et-st))
    
    # Depth scaling
    def scaling(self):
        print('Starting Scaling computation...')
        st = t.time()
        if np.min(self.crop_rect_left_MDE[:,200:500])<0.8 or np.min(self.crop_rect_right_MDE[:,140:440])<0.8:
            self.left_scaling_factor=1
            self.right_scaling_factor=1
            print("@@@")
        print("\t", np.min(self.crop_rect_left_MDE[:,200:500]), np.min(self.crop_rect_right_MDE[:,140:440]))
            
            
        self.scaled_left_MDE = self.rect_left_MDE * self.left_scaling_factor
        self.scaled_right_MDE = self.rect_right_MDE * self.right_scaling_factor
        print("\tScaling Factor : " + str(self.left_scaling_factor) + "  " + str(self.right_scaling_factor))
        et = t.time()
        print('\tScaling execution time \t\t\t\t= {:.3f}s'.format(et-st))
        
    
    def navi_preprocessing(self):
        print('Starting Navi\'s Preprocessing computation...')
        st = t.time()
        # Crop
        self.merge_rect_left_RGB = self.rect_left_RGB[105:375,:320]
        self.merge_rect_right_RGB = self.rect_right_RGB[105:375,320:]
        self.merge_rect_left_MDE = self.scaled_left_MDE[105:375,:320]
        self.merge_rect_right_MDE = self.scaled_right_MDE[105:375,320:]
        self.merge_left_stereo_depth = self.left_stereo_depth[105:375,:320]
        self.merge_right_stereo_depth = self.right_stereo_depth[105:375,320:]
        # Superpixel
        self.left_seg_center, __ = self.sp.superPixel(self.merge_rect_left_RGB, self.merge_left_stereo_depth, self.merge_rect_left_MDE)
        self.right_seg_center, __ = self.sp.superPixel(self.merge_rect_right_RGB, self.merge_right_stereo_depth, self.merge_rect_right_MDE)
        self.right_seg_center[:,0] = self.right_seg_center[:,0] + 320
        et = t.time()
        print('\tNavi\'s Preprocessing execution time \t\t= {:.3f}s'.format(et-st))    
    
    # Drone Navigation in 3D-space
    def navigation(self):
        print('Starting Navigation computation...')
        st = t.time()
        self.left_angular_velocity, self.thrust = self.navi.avoidObstacle2(self.left_seg_center)
        self.right_angular_velocity, self.thrust = self.navi.avoidObstacle2(self.right_seg_center)
        self.angular_velocity = self.left_angular_velocity + self.right_angular_velocity
        print(self.left_angular_velocity, self.right_angular_velocity)
        et = t.time()
        print('\tNavigation execution time \t\t\t= {:.3f}s'.format(et-st))
    
    def drone_ctrl(self):
        print('Starting Drone_Control computation...')
        st = t.time()
        self.drone.update_desired(self.angular_velocity, self.thrust)
        et = t.time()
        print('\tDrone_Command execution time \t\t\t= {:.3f}s'.format(et-st))
        
    
    def display(self):
        # Opencv Display : 0
        # Matplotlib Display : 1
        display_opt = 0
        
        # Opencv Display
        if display_opt == 0:    
            # RGB_viz = np.hstack((self.merge_rect_left_RGB,self.merge_rect_right_RGB))
            
            # __, left_MDE_viz = DISPLAY(self.merge_rect_left_MDE)
            # __, right_MDE_viz = DISPLAY(self.merge_rect_right_MDE)
            # MDE_viz = np.hstack((left_MDE_viz, right_MDE_viz))
            
            # cv2.imshow("1. RGB_viz", MDE_viz)
            # cv2.waitKey(10)
            
            
            MDE_viz = np.hstack((self.merge_rect_left_MDE, self.merge_rect_right_MDE))
            plt.imshow(MDE_viz)
            plt.show()            
            

        
        # Matplotlib Display  
        if display_opt == 1:
            # 1
            plt.subplot(4,2,1)
            plt.imshow(self.left_RGB)
            plt.title("Raw left image")
            plt.subplot(4,2,2)
            plt.imshow(self.right_RGB)
            plt.title("Raw right image")
            
            plt.subplot(4,2,3)
            plt.imshow(self.left_GT)
            plt.title("Left GT")
            plt.subplot(4,2,4)
            plt.imshow(self.right_GT)
            plt.title("Right GT")
            
            plt.subplot(4,2,5)
            plt.imshow(self.left_MDE)
            plt.title("Left MDE")
            plt.subplot(4,2,6)
            plt.imshow(self.right_MDE)
            plt.title("Right MDE")
            
            plt.subplot(4,2,7)
            plt.imshow(self.scaled_left_MDE)
            plt.title("Left MDE(scaled)")
            plt.subplot(4,2,8)
            plt.imshow(self.scaled_right_MDE)
            plt.title("Right MDE(scaled)")
            
            
            
            plt.show()
            
            # 2
            plt.subplot(3,2,1)
            plt.imshow(self.crop_rect_left_RGB)
            plt.title("left image")
            plt.subplot(3,2,2)
            plt.imshow(self.crop_rect_right_RGB)
            plt.title("right image")
            
            plt.subplot(3,2,3)
            plt.imshow(self.crop_rect_left_MDE)
            plt.title("left MDE")
            plt.subplot(3,2,4)
            plt.imshow(self.crop_rect_right_MDE)
            plt.title("right MDE")
            
            plt.subplot(3,2,5)
            plt.imshow(self.left_stereo_depth)
            plt.title("left SDE")
            plt.subplot(3,2,6)
            plt.imshow(self.right_stereo_depth)
            plt.title("right SDE")
            
            plt.show()
            
            
            plt.subplot(2,2,1)
            plt.imshow(self.merge_rect_left_RGB,)
            plt.subplot(2,2,2)
            plt.imshow(self.merge_rect_right_RGB)
            plt.subplot(2,2,3)
            plt.imshow(self.merge_rect_left_MDE)
            plt.subplot(2,2,4)
            plt.imshow(self.merge_rect_right_MDE)
            
            plt.show()
            
            
    def evaluation(self):
        
        e1 = evaluate(self.left_GT, self.left_MDE)
        e2 = evaluate(self.right_GT, self.right_MDE)
        e = (e1 + e2)
        
        print("\nBefore scaling")
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
        print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0]/2,e[1]/2,e[2]/2,e[3]/2,e[4]/2,e[5]/2))
        
        
        e1 = evaluate(self.left_GT, self.scaled_left_MDE)
        e2 = evaluate(self.right_GT, self.scaled_right_MDE)
        e = (e1 + e2)
        
        print("After scaling")
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
        print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0]/2,e[1]/2,e[2]/2,e[3]/2,e[4]/2,e[5]/2))
        

    
if __name__ == '__main__':
    try:
        ht = HumanTech()
           
        while not rospy.is_shutdown():
            '''
            "Common Part"
            '''
            ht.input_data()
            ht.MDE()
            ht.rectification()
            
            '''
            "Depth Scaling Part"
            '''
            ht.stereomathcing()
            ht.superpixel()
            ht.scaling()
            
            '''
            "Navigation Part"
            '''
            ht.navi_preprocessing()
            ht.navigation()
            ht.drone_ctrl()
            
            '''
            "Display"
            '''
            # ht.display()
            
            '''
            "Evaluation"
            '''
            # ht.evaluation()
            
            
    except rospy.ROSInterruptException:
        pass
    finally:
        print("\n\n// END //\n")
