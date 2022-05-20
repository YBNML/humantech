'''
#include "BehaviourArbitration.h"

The old Kinect has a depth image resolution of 320 x 240 pixels with a fov of 58.5 x 46.6
degrees resulting in an average of about 5 x 5 pixels per degree. (see source 1) << 1 patch = 5x5 pixel

source: http://smeenk.com/kinect-field-of-view-comparison/
'''

import cv2
import math
import numpy as np
import time as t

from numba import njit, objmode
from numba import int32, float64
from numba.experimental import jitclass

spec = [
    ('value', float64),               # a simple scalar field
    ('array', float64[:]),          # an array field
]
  
@jitclass(spec)
class Local_Path:
    def __init__(self):
        print("For obstacle avoidance")
        
    def navigation(self, seg_center):
        new_input = self.set_input(seg_center)
        
        L = new_input.shape[0]
        
        angleGain = 150
        colGain = 0.8
        hfov = 87
        vfov = 58
        angularRangeHorz = hfov * math.pi/180
        angularRangeVert = vfov * math.pi/180
        image_width = 640
        image_height = 400
        
        # init
        Horz_total = 0
        Vert_total = 0
        collision_prob = 0
        
        for i in range(L):
            # 장애물 회피를 목적으로, 거리가 가까울수록 영향력 큼
            distanceExponent = np.exp(-new_input[i,3])
            distanceExponent = distanceExponent * new_input[i,4] * L / image_height / image_width
            
            # View Angle (-FOV/2 < x < +FOV/2)
            # FOV와 좌표를 기반으로 이미지의 중심으로부터의 각을 구함
            Horz_Bearing = (hfov*new_input[i,0]/image_width) + (hfov/2/image_width)
            Vert_Bearing = (vfov*new_input[i,1]/image_height) + (vfov/2/image_height)
            # degree to radian
            Horz_Bearing = Horz_Bearing * math.pi / 180
            Vert_Bearing = Vert_Bearing * math.pi / 180
            
            # The larger the center, the smaller the border.
            Horz_bearingWeight = np.exp(-pow(Horz_Bearing,2)/(pow(angularRangeHorz,2)))
            Vert_bearingWeight = np.exp(-pow(Vert_Bearing,2)/(pow(angularRangeVert,2)))
            
            Horz_total += (Horz_Bearing * Horz_bearingWeight) * distanceExponent   
            Vert_total += (Vert_Bearing * Vert_bearingWeight) * distanceExponent
            
            collision_prob += (Horz_bearingWeight * distanceExponent) + (Vert_bearingWeight * distanceExponent)
        
        collision_prob = collision_prob/L
        collision_prob = collision_prob * np.exp(pow(collision_prob*colGain,2))
        if collision_prob > 1:
            collision_prob = 1
             
        Horz_total = Horz_total * np.exp(-pow(Horz_total/angleGain,2))
        if Horz_total > hfov:
            Horz_total = hfov
        elif Horz_total < -hfov:
            Horz_total = -hfov
        
        Vert_total = Vert_total * np.exp(-pow(Vert_total/angleGain,2))
        if Vert_total > vfov:
            Vert_total = vfov
        elif Vert_total < -vfov:
            Vert_total = -vfov
        Vert_total = Vert_total/10
        
        
        # print(Horz_total)
        # # print(Vert_total)
        # print(collision_prob)
        return Horz_total, Vert_total, collision_prob
    

    def set_input(self, seg_center):
        L = seg_center.shape[0]
        new_input = np.zeros((L,5))

        for i in range(L):
            x_dis = seg_center[i,0] - 320
            y_dis = seg_center[i,1] - 240
            
            new_input[i,0] = x_dis                      # x_pos
            new_input[i,1] = y_dis                      # y_pos
            
            new_input[i,2] = pow(x_dis,2) + pow(y_dis,2)
            new_input[i,2] = math.sqrt(new_input[i,2])  # Distance[pixel] from center of image

            new_input[i,3] = seg_center[i,3]            # Depth
            new_input[i,4] = seg_center[i,4]            # The number of segmentation's pixels
        
        return new_input

if __name__ == "__main__":
    print("test")
    Ld=np.load("left_data.npy")
    Rd=np.load("right_data.npy")
    data = np.vstack((Ld,Rd))
    # print(data)
    local_p = Local_Path()
    
    st = t.time()
    for i in range(1):
        local_p.navigation(data)
    print(t.time()-st)