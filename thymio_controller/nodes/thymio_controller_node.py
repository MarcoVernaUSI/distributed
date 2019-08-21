#!/usr/bin/env python

# import os
# from math import copysign, cos, log, sin, sqrt
 
import roslib
import rospy
import sys
import torch
from network import DistributedNet
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range
# import std_srvs.srv
# from asebaros_msgs.msg import AsebaEvent
# from asebaros_msgs.srv import GetNodeList, LoadScripts
# from dynamic_reconfigure.server import Server
# from geometry_msgs.msg import Quaternion, Twist
# from nav_msgs.msg import Odometry
# from sensor_msgs.msg import Imu, JointState, Joy, LaserScan, Range, Temperature
# from std_msgs.msg import Bool, ColorRGBA, Empty, Float32, Int8, Int16
# from tf.broadcaster import TransformBroadcaster
# from thymio_driver.cfg import ThymioConfig
# from thymio_msgs.msg import Led, LedGesture, Sound, SystemSound, Comm

BASE_WIDTH = 91.5     # millimeters
MAX_SPEED = 500.0     # units
SPEED_COEF = 2.93     # 1mm/sec corresponds to X units of real thymio speed
WHEEL_RADIUS = 0.022   # meters
GROUND_MIN_RANGE = 9     # millimeters
GROUND_MAX_RANGE = 30     # millimeters

PROXIMITY_NAMES = ['left', 'center_left', 'center',
				   'center_right', 'right', 'rear_left', 'rear_right']

sensor_max_range = 0.14
sensor_min_range = 0.0215
max_vel = 0.14


class ThymioController(object):

#    def frame_name(self, name):
#        if self.tf_prefix:
#            return '{self.tf_prefix}/{name}'.format(**locals())
#        return name


	def __init__(self):
		rospy.init_node('thymio_controller')
		self.thymio_name=rospy.get_namespace()[1:]
		self.state = np.zeros(2) #distanza avanti e dietro (if 2) # velocit dei vicini se 4
		

		# Carico rete    
		net_file = rospy.get_param('net')
		self.net = torch.load(str(net_file))

		# Inizializzo tutto

		# Subscriber e publiscer
			# Publish to the topic '/thymioX/cmd_vel'.  # forse serve la / iniziale
		self.velocity_publisher = rospy.Publisher('cmd_vel',Twist, queue_size=10)

		# Subscribers to sensors
		self.sensor_subscribers=[
		rospy.Subscriber('proximity/'+name, Range, self.update_range)
		for name in PROXIMITY_NAMES]

		self.current_twist = Twist()
		self.current_range = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

		# publish at this rate
		self.rate = rospy.Rate(10)
		
	def update_range(self, data):

#np.clip(data.range, sensor_min_range, sensor_max_range)
		# update range state any time a message is received
		# min_range, max_range
		if data.header.frame_id == self.thymio_name + "proximity_center_link":
			self.current_range[0] = np.clip(data.range, sensor_min_range, sensor_max_range)
		elif data.header.frame_id == self.thymio_name + "proximity_center_left_link":
			self.current_range[1] = np.clip(data.range, sensor_min_range, sensor_max_range)
		elif data.header.frame_id == self.thymio_name + "proximity_center_right_link":
			self.current_range[2] = np.clip(data.range, sensor_min_range, sensor_max_range)
		elif data.header.frame_id == self.thymio_name + "proximity_left_link":
			self.current_range[3] = np.clip(data.range, sensor_min_range, sensor_max_range)
		elif data.header.frame_id == self.thymio_name + "proximity_right_link":
			self.current_range[4] = np.clip(data.range, sensor_min_range, sensor_max_range)
		elif data.header.frame_id == self.thymio_name + "proximity_rear_left_link":
			self.current_range[5] = np.clip(data.range, sensor_min_range, sensor_max_range)
		elif data.header.frame_id == self.thymio_name + "proximity_rear_right_link":
			self.current_range[6] = np.clip(data.range, sensor_min_range, sensor_max_range)

		self.update_sensing()
	

	def update_sensing(self):
		self.state= np.array([np.min(self.current_range[0:5]),np.min(self.current_range[5:])])


	def move(self, seconds):

		vel_msg = Twist()
		vel_msg.angular.z = 0.0 # rad/s

		t = 0

		while t < seconds*10:
			#print(self.thymio_name + str(self.state))
			vel_msg.linear.x =  np.clip(- self.step(self.state,self.net), -max_vel, max_vel)# m/s
			self.velocity_publisher.publish(vel_msg)
			self.rate.sleep()
			t=t+1

	def stop(self):

		vel_msg = Twist()
		vel_msg.linear.x = 0. # m/s
		vel_msg.angular.z = 0. # rad/s



		self.velocity_publisher.publish(vel_msg)
		self.rate.sleep()

	def step(self, sensing, net):
		net_controller = net.controller()
		control =net_controller(sensing)
		return control[0]


if __name__ == '__main__':

	controller = ThymioController()

	controller.move(30)
	controller.stop()


		# Wait for sincronization

	#    rospy.wait_for_service('aseba/get_node_list')
	#    get_aseba_nodes = rospy.ServiceProxy(
	#        'aseba/get_node_list', GetNodeList)

		# Agisco
	rospy.spin()


