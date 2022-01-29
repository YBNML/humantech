'''
#include "BehaviourArbitration.h"

The old Kinect has a depth image resolution of 320 x 240 pixels with a fov of 58.5 x 46.6
degrees resulting in an average of about 5 x 5 pixels per degree. (see source 1) << 1 patch = 5x5 pixel

source: http://smeenk.com/kinect-field-of-view-comparison/
'''

import math
import numpy as np
import cv2

from numba import njit

class Navigation:
    def __init__(self):
        # Goal = goto
        self.lambdaObstacleHorzNormal = 5;           # 장애물 noraml??, 장애물 인지 계수
        self.lambdaObstacleHorz = self.lambdaObstacleHorzNormal # Default value
        
        self.lambdaObstacleVert = 5
        
        # w, 3페이지에 4번 수식 참고
        self.weightGoalHorz = 0.3
        self.weightObstacleHorz = 0.7
        
        # gain 설정
        self.obstacleDistanceGainHorzNormal = 0.1            # Smaller = more sensitive 
        self.obstacleDistanceGainHorz = self.obstacleDistanceGainHorzNormal  # Default value
        
        self.hfov = 112
        self.vfov = 50
        self.angularRangeHorz = (self.hfov)*math.pi/180
        self.angularRangeVert = (self.vfov)*math.pi/180
        
        self.image_height = 270
        self.image_width  = 640
        
        
    def avoidObstacle(self, depth_image):
        depth = depth_image[135,:]
        Horz_fObstacleTotal = 0
        
        for i in range(self.image_width):
            Horz_obstacleBearing = (self.hfov*i/self.image_width) - (self.hfov/2) + (self.hfov/2/self.image_width)
            Horz_obstacleBearing = Horz_obstacleBearing * math.pi / 180
            
            Horz_bearingExponent = np.exp(-pow(Horz_obstacleBearing,2)/(2*pow(self.angularRangeHorz,2)))
            
            distanceExponent = np.exp(-self.obstacleDistanceGainHorz * depth[i]) 
            
            Horz_fObstacleTotal += Horz_obstacleBearing * Horz_bearingExponent * distanceExponent
        
        angular_velocity = Horz_fObstacleTotal * self.lambdaObstacleHorz / 10
        
        return angular_velocity
    
    def avoidObstacle2(self, seg_center):
        self.seg_center = seg_center
        
        L = self.seg_center.shape[0]
        Horz_fObstacleTotal = 0
        Vert_fObstacleTotal = 0
    
        self.detectCorner()
        
        for i in range(L):
            if self.seg_center[i,3]>20:
                self.seg_center[i,3] = 20
            # print(self.seg_center[i])
            # View Angle (-FOV/2 < x < +FOV/2)
            Horz_obstacleBearing = (self.hfov*self.seg_center[i,0]/self.image_width) - (self.hfov/2) + (self.hfov/2/self.image_width)
            Vert_obstacleBearing = (self.vfov*self.seg_center[i,1]/self.image_height) - (self.vfov/2) + (self.vfov/2/self.image_height)
        
        
            # degree to radian
            Horz_obstacleBearing = Horz_obstacleBearing * math.pi / 180
            Vert_obstacleBearing = Vert_obstacleBearing * math.pi / 180
        
            # The larger the center, the smaller the border.
            Horz_bearingExponent = np.exp(-1*pow(Horz_obstacleBearing,2)/(2*pow(self.angularRangeHorz,2)))
            Vert_bearingExponent = np.exp(-1*pow(Vert_obstacleBearing,2)/(2*pow(self.angularRangeVert,2)))

            distanceExponent = np.exp(-self.obstacleDistanceGainHorz * self.seg_center[i,3]*5) 

            Horz_fObstacleTotal += Horz_obstacleBearing * Horz_bearingExponent * distanceExponent * self.seg_center[i,4]
            Vert_fObstacleTotal += Vert_obstacleBearing * Vert_bearingExponent * distanceExponent * self.seg_center[i,4]
        
        # 단위는 degree
        angular_velocity = Horz_fObstacleTotal * self.lambdaObstacleHorz / self.image_height / 10
        Thrust = Vert_fObstacleTotal * self.lambdaObstacleVert / self.image_width / 100
    
        return angular_velocity, Thrust, 0.5
    
    
    def detectCorner(self):
        L = self.seg_center.shape[0]
        detectCorner_pnt = 0
        
        for i in range(L):
            xpos = self.seg_center[i,0] - 319.5
            detectCorner_coeff = -0.00008789*(xpos-543.89)*(xpos+543.89)
            detectCorner_pnt += detectCorner_coeff * self.seg_center[i,3] * self.seg_center[i,4]
            
        detectCorner_pnt = detectCorner_pnt / self.image_height / self.image_width * 7
        
        # self.lambdaObstacleHorz = 1600 * pow(detectCorner_pnt,-0.964)
        self.lambdaObstacleHorz = 250 * pow(detectCorner_pnt,-0.6)
        
        self.obstacleDistanceGainHorz = 0.0368 * np.log(detectCorner_pnt) - 0.125
        
            
        
        