#!/usr/bin/env python3

import rospy
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image

import numpy as np

# Global Variable
left_color  = np.zeros((480,640,3), dtype=np.uint8)
right_color = np.zeros((480,640,3), dtype=np.uint8)
left_depth  = np.zeros((480,640,1), dtype=np.float64)
right_depth = np.zeros((480,640,1), dtype=np.float64)

# Subscribe topic(RGB iamge) 
def callback_left_Color(data):
    global left_color
    left_color = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, 3)
# Subscribe topic(RGB iamge) 
def callback_right_Color(data):
    global right_color
    right_color = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, 3)
# Subscribe topic(Depth iamge) 
def callback_left_Depth(data):
    global left_depth
    left_depth = np.frombuffer(data.data, dtype=np.float32).reshape(data.height, data.width, -1)
# Subscribe topic(Depth iamge) 
def callback_right_Depth(data):
    global right_depth
    right_depth = np.frombuffer(data.data, dtype=np.float32).reshape(data.height, data.width, -1)


'''
Received image from 'rotors(gazebo)'.
'''
class Image_load():
    def __init__(self):
        # ROS
        rospy.init_node('HumanTech_NODE', anonymous=True)
        rospy.loginfo("Waiting for HumanTech_NODE")
        r=rospy.Rate(20)    

        # Subscribe Node
        rospy.Subscriber("/firefly/left_cam_RGBD/RGB", Image, callback_left_Color)
        rospy.Subscriber("/firefly/right_cam_RGBD/RGB", Image, callback_right_Color)
        rospy.Subscriber("/firefly/left_cam_RGBD/Depth", Image, callback_left_Depth)
        rospy.Subscriber("/firefly/right_cam_RGBD/Depth", Image, callback_right_Depth)
        
    # Load RGB image
    def ROS_RGB(self):
        global left_color, right_color
        return left_color, right_color

    # Load Ground_Truth depth image
    def ROS_GT(self):
        global left_depth, right_depth
        return left_depth, right_depth
        





    # def load_rgb(self):
    #     DIR = str(ROOT) + "/dataset/rgb/"
    #     left_rgb = cv2.imread(DIR+"left_color"+str(self.count)+".npy")
    #     right_rgb = cv2.imread(DIR+"right_color"+str(self.count)+".npy")
    #     return left_rgb, right_rgb

    # def load_gt(self):
    #     DIR = str(ROOT) + "/dataset/gt/"
    #     left_gt = cv2.imread(DIR+"left_depth"+str(self.count)+".npy")
    #     right_gt = cv2.imread(DIR+"right_depth"+str(self.count)+".npy")
    #     return left_gt, right_gt

    # def load_mde(self):
    #     mde = "adabins"     # choose MDE model (adabins, bts, densedepth)
    #     DIR = str(ROOT) + "/dataset/" + mde +"/"
    #     left_mde = cv2.imread(DIR + "left_" + mde + str(self.count) + ".npy")
    #     right_mde = cv2.imread(DIR + "right_" + mde + str(self.count) + ".npy")
    #     return left_mde, right_mde