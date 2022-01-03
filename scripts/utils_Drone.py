#!/usr/bin/env python3

import rospy
import tf
import std_msgs.msg
from geometry_msgs.msg import Twist, Transform, PointStamped
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint


class Drone_CTRL():
    def __init__(self):
        # rospy.init_node('Drone_Control')
        pub = rospy.Publisher('/firefly/command/trajectory', MultiDOFJointTrajectory, queue_size=1)
        # sub = rospy.Subscriber('/cluster_decomposer/boxes', BoundingBoxArray, cb, queue_size=1)
        # rospy.spin()
        
        self.traj_msg = MultiDOFJointTrajectory()
        self.point_msg = MultiDOFJointTrajectoryPoint()
        
        self.header = std_msgs.msg.Header()
        self.header.frame_id = 'frame'
        # self.quaternion = tf.transformations.quaternion_from_euler(0,0,0)
        # self.traj_msg.joint_names.clear()
        # self.traj_msg.joint_names = "Rotors"
        
        
        # self.traj_msg.points
        
        # replan_pos = PointStamped()
        # self.velocities = Twist()
        # self.accelerations = Twist()
        
    def publish(self, yaw, thrust):
        print(yaw)
        print(thrust)
        # print("test")
        
        
        
#         def cb(msg):
#         filtered_boxes = BoundingBoxArray()
#     filtered_boxes.header = msg.header
#     for box in msg.boxes:
#         if box.dimensions.x==box.dimensions.y==box.dimensions.z==0.0:
#             continue
#         print("dimensions: \n",box.dimensions)
#         # if(box.dimensions.x)
#         if(box.dimensions.x>1):
#             box.dimensions.x = box.dimensions.x/2
#         if(box.dimensions.y>1):
#             box.dimensions.y = box.dimensions.y/2
#         if(box.dimensions.z>1):
#             box.dimensions.z = box.dimensions.z/2
#         if(box.dimensions.x>2 or box.dimensions.y>2 or box.dimensions.z>2):
#             box.dimensions.x = box.dimensions.y =box.dimensions.z =0
#         # seed random number generator
#         seed(1)
#         # generate some integers
#         box.value = randint(0, 10)
#         print(box.value)
#         filtered_boxes.boxes.append(box)
#     print(filtered_boxes.boxes.__len__())

#     # pc = pcl.PointCloud(10)
#     # a = np.asarray(pc)
#     # a[:] = 0
#     # a[:,0] = filtered_boxes.boxes[0].pose.position.x
#     # a[:,1] = filtered_boxes.boxes[0].pose.position.y
#     # a[:,2] = filtered_boxes.boxes[0].pose.position.z
#     # new_cloud = pcl.PointCloud()
#     # new_cloud.from_array(a)
#     # new_cloud = pcl_helper.XYZ_to_XYZRGB(new_cloud,[255,255,255])

#     pub.publish(filtered_boxes)
#     # pub.publish(new_cloud)
# if __name__ == '__main__':
#     rospy.init_node('box_to_pcl')
#     pub = rospy.Publisher('/boxes_filtered', BoundingBoxArray, queue_size=1)
#     # pub = rospy.Publisher('PL_filtered', PointCloud2, queue_size=1)
#     sub = rospy.Subscriber('/cluster_decomposer/boxes', BoundingBoxArray, cb, queue_size=1)
#     # sub = rospy.Subscriber('/cluster_decomposer/boxes', BoundingBoxArray, cb, queue_size=1)
    
#     rospy.spin()