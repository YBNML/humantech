#!/usr/bin/env python3

import numpy as np
import cv2

from matplotlib import pyplot as plt
from utils_Parameter import parameter
# from utils_Deproject import deproject_pixel_to_point
# from utils_Rotation import Rotation

#####################################################
##              Rectification                      ##
#####################################################


class Rectification:
    def __init__(self):
        R1 = [[0.96592582628, 0.0, -0.2588190451], [0.0, 1.0, 0.0], [0.2588190451, 0.0, 0.96592582628]]
        t1 = [[-0.1],[0],[0]]
        t1 = np.array(t1)

        R2 = [[0.96592582628, 0.0, 0.2588190451], [0.0, 1.0, 0.0], [-0.2588190451, 0.0, 0.96592582628]]
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


    # def remap_data(self, left_img, right_img):
    #     # Eval
    #     H,W = left_img.shape
    #     for ih in range(H):
    #         for iw in range(W):
    #             x1,y1,z1 = deproject_pixel_to_point(iw, ih, left_img[ih, iw])
    #             x2,y2,z2 = deproject_pixel_to_point(iw, ih, right_img[ih, iw])
    #             x1, z1 = Rotation(x1, z1, degree=15, up=True)
    #             x2, z2 = Rotation(x2, z2, degree=15, up=False)
    #             left_img[ih, iw] = z1
    #             right_img[ih, iw] = z2
    #     return left_img, right_img



# class Rectification2:
#     def __init__(self):
#         R1 = [[0.96592582628, 0.0, -0.2588190451], [0.0, 1.0, 0.0], [0.2588190451, 0.0, 0.96592582628]]
#         t1 = [[-0.1],[0],[0]]
#         t1 = np.array(t1)

#         R2 = [[0.96592582628, 0.0, 0.2588190451], [0.0, 1.0, 0.0], [-0.2588190451, 0.0, 0.96592582628]]
#         t2 = [[0.1],[0],[0]]
#         t2 = np.array(t2)
        
#         R = np.matmul(np.linalg.inv(R2), R1)
#         T = np.matmul(np.linalg.inv(R2), (t1 - t2))
        
#         param = parameter()
#         px = param.get_px()
#         py = param.get_py()
#         fx = param.get_fx()
#         fy = param.get_fy()

#         cameraMatrix1 = np.array(
#             [
#                 [fx, 0, px],
#                 [0, fy, py],
#                 [0, 0, 1.0]
#             ]
#         )
#         cameraMatrix2 = cameraMatrix1
#         distCoeff = np.zeros(4)

#         imgSize = (param.get_img_w(), param.get_img_h())

#         R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
#             cameraMatrix1=cameraMatrix1,
#             distCoeffs1=distCoeff,
#             cameraMatrix2=cameraMatrix2,
#             distCoeffs2=distCoeff,
#             imageSize=imgSize,
#             R=R,
#             T=T,
#             flags=cv2.CALIB_ZERO_DISPARITY,
#             alpha=1)

#         self.map1x, self.map1y = cv2.initUndistortRectifyMap(
#             cameraMatrix=cameraMatrix1,
#             distCoeffs=distCoeff,
#             R=R1,
#             newCameraMatrix=P1,
#             size=imgSize,
#             m1type=cv2.CV_32FC1)

#         self.map2x, self.map2y = cv2.initUndistortRectifyMap(
#             cameraMatrix=cameraMatrix2,
#             distCoeffs=distCoeff,
#             R=R2,
#             newCameraMatrix=P2,
#             size=imgSize,
#             m1type=cv2.CV_32FC1)


#     def remap_img(self, left_img, right_img):
#         img1_rect = cv2.remap(left_img, self.map1x, self.map1y, cv2.INTER_LINEAR)
#         img2_rect = cv2.remap(right_img, self.map2x, self.map2y, cv2.INTER_LINEAR)
#         return img1_rect, img2_rect


#     def remap_data(self, left_img, right_img):
#         # Eval
#         H,W = left_img.shape
#         for ih in range(H):
#             for iw in range(W):
#                 x1,y1,z1 = deproject_pixel_to_point(iw, ih, left_img[ih, iw])
#                 x2,y2,z2 = deproject_pixel_to_point(iw, ih, right_img[ih, iw])
#                 x1, z1 = Rotation(x1, z1, degree=15, up=True)
#                 x2, z2 = Rotation(x2, z2, degree=15, up=False)
#                 left_img[ih, iw] = z1
#                 right_img[ih, iw] = z2
#         return left_img, right_img