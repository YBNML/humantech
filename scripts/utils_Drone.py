#!/usr/bin/env python3

import math

import rospy
import tf
import std_msgs.msg
from geometry_msgs.msg import Twist, Transform, Quaternion, Point, Pose
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint


class Drone_CTRL():
    def __init__(self):
        # rospy.init_node('Drone_Control')
        self.pub = rospy.Publisher('/firefly/command/trajectory', MultiDOFJointTrajectory, queue_size=1)
        rospy.Subscriber('/firefly/ground_truth/pose', Pose, self.update_pose, queue_size=1)
        # rospy.spin()
        
        self.msg = MultiDOFJointTrajectory()
        
        self.header = std_msgs.msg.Header()
        self.header.frame_id = 'frame'
        
        self.quaternion = tf.transformations.quaternion_from_euler(0,0,0)
        self.velocities = Twist()
        self.accelerations = Twist()
        
        # Init parameter
        self.current_x = 0
        self.current_y = 0
        self.current_z = 1
        self.velocity = 0.5
        self.desired_yaw = 0
        
        self.pub_flag = False
        
        

    def update_pose(self, data):
        self.current_x = data.position.x
        self.current_y = data.position.y
        self.current_z = data.position.z
        
        quaternion = (
            data.orientation.x,
            data.orientation.y,
            data.orientation.z,
            data.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.current_yaw = euler[2]
        
        if self.pub_flag:
            self.header.stamp = rospy.Time()
            self.msg.header = self.header
            
            self.quaternion = tf.transformations.quaternion_from_euler(0,0,self.current_yaw)
            transforms = Transform(translation=Point(self.current_x,self.current_y,1), 
                                   rotation=Quaternion(self.quaternion[0],self.quaternion[1],self.quaternion[2],self.quaternion[3]))

            self.velocities.linear.x = self.velocity*math.cos(self.current_yaw)
            self.velocities.linear.y = self.velocity*math.sin(self.current_yaw)
            self.velocities.linear.z = self.desired_thrust
            self.velocities.angular.z = self.desired_yaw
            

            p = MultiDOFJointTrajectoryPoint([transforms], [self.velocities], [self.accelerations], rospy.Time(rospy.get_time()))
            self.msg.points.append(p) 
            
            self.pub.publish(self.msg)
            self.msg.points.clear()
        
        
    def update_ours(self, yaw, thrust, collision_prob):
        yaw = 1 * yaw * math.pi / 180      # degree to radian
        self.desired_yaw = yaw
        
        
        self.desired_thrust = thrust * pow((1-collision_prob),2)
        self.desired_thrust = -thrust
        if(collision_prob > 0.7):  self.desired_thrust = 0
        self.desired_thrust = 0
        
        self.velocity = 1 * pow((1-collision_prob),2)
        if(collision_prob > 0.7):  self.velocity = 0
        print(self.velocity)
        
        print("\t", self.desired_yaw, self.desired_thrust, collision_prob)
        
        self.pub_flag = True
        
        
    def update_dronet(self, velocity, steering):
        self.velocity = velocity * 1.2
        self.desired_yaw = -steering
        self.desired_thrust = 0
        
        self.pub_flag = True
        
        
        
        