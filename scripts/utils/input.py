#!/usr/bin/env python3

import rospy
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image

import cv2
import numpy as np

from pathlib import Path
FILE = Path(__file__).resolve()
Dataset_WS = FILE.parents[0]

# Global Variable
left_color  = np.zeros((480,640,3), dtype=np.uint8)
right_color = np.zeros((480,640,3), dtype=np.uint8)
center_color = np.zeros((480,640,3), dtype=np.uint8)

left_depth  = np.zeros((480,640), dtype=np.float32)
right_depth = np.zeros((480,640), dtype=np.float32)
center_depth = np.zeros((480,640), dtype=np.float32)

# Subscribe topic(left iamge) 
def callback_left_Color(data):
    global left_color
    left_color = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, 3)
def callback_left_Depth(data):
    global left_depth
    left_depth = np.frombuffer(data.data, dtype=np.float32).reshape(data.height, data.width, -1)
    
# Subscribe topic(right iamge) 
def callback_right_Color(data):
    global right_color
    right_color = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, 3)
def callback_right_Depth(data):
    global right_depth
    right_depth = np.frombuffer(data.data, dtype=np.float32).reshape(data.height, data.width, -1)
    
# Subscribe topic(center iamge) 
def callback_center_Color(data):
    global center_color
    center_color = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, 3)
def callback_center_Depth(data):
    global center_depth
    center_depth = np.frombuffer(data.data, dtype=np.float32).reshape(data.height, data.width, -1)
    
    


'''
Received image from 'rotors(gazebo)'.
'''
class Image_load():
    def __init__(self):
        # ROS
        rospy.init_node('HumanTech_NODE', anonymous=True)
        rospy.loginfo("Waiting for HumanTech_NODE")
        # r=rospy.Rate(20)    

        # Subscribe Node
        rospy.Subscriber("/firefly/left_cam_RGBD/RGB", Image, callback_left_Color)
        rospy.Subscriber("/firefly/right_cam_RGBD/RGB", Image, callback_right_Color)
        rospy.Subscriber("/firefly/center_cam_RGBD/RGB", Image, callback_center_Color)
        rospy.Subscriber("/firefly/left_cam_RGBD/Depth", Image, callback_left_Depth)
        rospy.Subscriber("/firefly/right_cam_RGBD/Depth", Image, callback_right_Depth)
        rospy.Subscriber("/firefly/center_cam_RGBD/Depth", Image, callback_center_Depth)
        
        # opt
        self.count = 0
        
        
    '''
    For our navigation algorithm
    '''
    def ROS_RGB(self):
        global left_color, right_color
        return left_color, right_color

    # Load Ground_Truth depth image
    def ROS_GT(self):
        global left_depth, right_depth
        left_depth = np.nan_to_num(left_depth, copy=True)
        right_depth = np.nan_to_num(right_depth, copy=True)
        return left_depth, right_depth
    
    
    '''
    For other navigation algorithm
    '''
    def ROS_RGB2(self):
        global center_color
        return center_color
    def ROS_GT2(self):
        global center_depth
        center_depth = np.nan_to_num(center_depth, copy=True)
        return center_depth
    
    
    
    
    def test_RGB(self):
        DIR             = str(Dataset_WS) + "/dataset/rgb/"
        
        temp_left_rgb   = np.load(DIR+"left_color"+str(self.count)+".npy")
        temp_left_rgb   = cv2.resize(temp_left_rgb, dsize=(960, 720), interpolation=cv2.INTER_LINEAR)
        left_rgb        = np.zeros((720, 1280 ,3), dtype=np.uint8)
        left_rgb[:, 160:1120,:] = temp_left_rgb
        
        temp_right_rgb  = np.load(DIR+"right_color"+str(self.count)+".npy")
        temp_right_rgb  = cv2.resize(temp_right_rgb, dsize=(960, 720), interpolation=cv2.INTER_LINEAR)
        right_rgb       = np.zeros((720, 1280 ,3), dtype=np.uint8)
        right_rgb[:, 160:1120,:] = temp_right_rgb
        
        return left_rgb, right_rgb
        
    def test_GT(self):
        DIR             = str(Dataset_WS) + "/dataset/gt/"
        
        temp_left_gt   = np.load(DIR+"left_depth"+str(self.count)+".npy")
        temp_left_gt   = cv2.resize(temp_left_gt, dsize=(960, 720), interpolation=cv2.INTER_LINEAR)
        left_gt        = np.zeros((720, 1280 ,3))
        left_gt[:, 160:1120,:] = temp_left_gt
        
        temp_right_gt  = np.load(DIR+"right_depth"+str(self.count)+".npy")
        temp_right_gt  = cv2.resize(temp_right_gt, dsize=(960, 720), interpolation=cv2.INTER_LINEAR)
        right_gt       = np.zeros((720, 1280 ,3))
        right_gt[:, 160:1120,:] = temp_right_gt
        
        self.count += 1
        return left_gt[:,:,2], right_gt[:,:,2]
        


class Realsense_Image():
    def __init__(self):
        # ROS
        rospy.init_node('IASL_KHJ', anonymous=True)
        
        # Subscribe Node
        rospy.Subscriber("/camera_left/color/image_raw", Image, callback_left_Color)
        rospy.Subscriber("/camera_right/color/image_raw", Image, callback_right_Color)
        
        
    def RS_RGB(self):
        global left_color, right_color
        return left_color, right_color
    
    
    def test(self):
        global left_color, right_color
        left_color = cv2.imread("left_Color.png")
        right_color = cv2.imread("right_Color.png")
        return left_color, right_color
        
