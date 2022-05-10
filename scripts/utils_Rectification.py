#!/usr/bin/env python3

import cv2
import math
import numpy as np

from matplotlib import pyplot as plt
from utils_Parameter import parameter
# from utils_Deproject import deproject_pixel_to_point
# from utils_Rotation import Rotation

#####################################################
##              Rectification                      ##
#####################################################


class Rectification:
    def __init__(self, degree=30):
        degree = degree/2
        radian = degree*math.pi/180
        
        cos_v = math.cos(radian)
        sin_v = math.sin(radian)
        
        R1 = [[cos_v, 0.0, -sin_v], [0.0, 1.0, 0.0], [sin_v, 0.0, cos_v]]
        t1 = [[-0.1],[0],[0]]
        t1 = np.array(t1)

        R2 = [[cos_v, 0.0, sin_v], [0.0, 1.0, 0.0], [-sin_v, 0.0, cos_v]]
        t2 = [[0.1],[0],[0]]
        t2 = np.array(t2)
        
        R = np.matmul(np.linalg.inv(R2), R1)
        T = np.matmul(np.linalg.inv(R2), (t1 - t2))
        
        param = parameter()
        px = param.get_px()
        py = param.get_py()
        fx = param.get_fx()
        fy = param.get_fy()

        cameraMatrix1 = np.array(
            [
                [fx, 0, px],
                [0, fy, py],
                [0, 0, 1.0]
            ]
        )
        cameraMatrix2 = cameraMatrix1
        distCoeff = np.zeros(4)

        imgSize = (param.get_img_w(), param.get_img_h())

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            cameraMatrix1=cameraMatrix1,
            distCoeffs1=distCoeff,
            cameraMatrix2=cameraMatrix2,
            distCoeffs2=distCoeff,
            imageSize=imgSize,
            R=R,
            T=T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=-1)

        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            cameraMatrix=cameraMatrix1,
            distCoeffs=distCoeff,
            R=R1,
            newCameraMatrix=P1,
            size=imgSize,
            m1type=cv2.CV_32FC1)

        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            cameraMatrix=cameraMatrix2,
            distCoeffs=distCoeff,
            R=R2,
            newCameraMatrix=P2,
            size=imgSize,
            m1type=cv2.CV_32FC1)


    def remap_img(self, left_img, right_img):
        img1_rect = cv2.remap(left_img, self.map1x, self.map1y, cv2.INTER_LINEAR)
        img2_rect = cv2.remap(right_img, self.map2x, self.map2y, cv2.INTER_LINEAR)
        return img1_rect, img2_rect