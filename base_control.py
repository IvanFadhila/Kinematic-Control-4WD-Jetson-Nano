#!/usr/bin/env python

import rospy
import numpy as np
import random
import tf
from numpy.linalg import inv
from geometry_msgs.msg import Vector3, Twist, Quaternion, PoseWithCovarianceStamped
from std_msgs.msg import Int32MultiArray, Float32MultiArray, Int32, Empty

class Server:
	def __init__(self):
		self.node_name = rospy.get_name() # membaca nama node yang dijalankan
		self.num_of_act = rospy.get_param(self.node_name+"/number_of_actuator",4)
		self.r = rospy.get_param(self.node_name+"/wheel_radius", 0.05)		
		if(self.num_of_act == 4):
			self.alp = rospy.get_param(self.node_name+"/alpha", [135., 45., -135., -45])
			self.gamma = rospy.get_param(self.node_name+"/gamma", [45., -45., 135., 135])
			self.l = rospy.get_param(self.node_name+"/wheels_distance", [0.215, 0.215, 0.215, 0.215])
			for i in range(len(self.alp)):
				self.alp[i] = np.radians(self.alp[i])
				self.gamma[i] = np.radians(self.gamma[i])
			self.Jr = self.get_invers_jacobian4()
		else:
			self.alp = rospy.get_param(self.node_name+"/alpha", [150., 30., -90.])
			self.gamma = rospy.get_param(self.node_name+"/gamma", [60., -60., 180.])
			self.l = rospy.get_param(self.node_name+"/wheels_distance", [0.215, 0.215, 0.215])
			for i in range(len(self.alp)):
				self.alp[i] = np.radians(self.alp[i])
				self.gamma[i] = np.radians(self.gamma[i])
			self.Jr = self.get_invers_jacobian3()
		self.max_rpm = rospy.get_param(self.node_name+"/max_rpm", 500.)
		self.min_rpm = rospy.get_param(self.node_name+"/min_rpm", 60.)
		self.cut_off_rpm = rospy.get_param(self.node_name+"/cut_off_rpm", 0.4)
		self.cmd_vel_topic = rospy.get_param(self.node_name+"/cmd_vel_topic", "/cmd_vel")
		self.motor_topic = rospy.get_param(self.node_name+"/motor_topic", "/pwm")
		self.screen_output = rospy.get_param(self.node_name+"/output_screen", False)
		self.pwm_publisher = rospy.Publisher(self.motor_topic, Quaternion, queue_size = 3)		
		self.delay = rospy.Rate(1)
		self.config_print()

	def config_print(self):
		print "============================================="
		print "Robot Config"
		print "============================================="
		print "Number of Actuators: ", self.num_of_act
		if(self.num_of_act == 4):
			print "Alpha Config:"
			print "     W1:", np.degrees(self.alp[0]), " W2:", np.degrees(self.alp[1]), " W3:", np.degrees(self.alp[2]), " W4:", np.degrees(self.alp[3])
			print "Gamma Config:"
			print "     W1:", np.degrees(self.gamma[0]), " W2:", np.degrees(self.gamma[1]), " W3:", np.degrees(self.gamma[2]), " W4:", np.degrees(self.gamma[3])
			print "Wheel Distance Config:"
			print "     l1:", self.l[0]," l2:", self.l[1]," l3:", self.l[2]," l4:", self.l[3]
		else:
			print "Alpha Config:"
			print "     W1:", np.degrees(self.alp[0])," W2:", np.degrees(self.alp[1])," W3:", np.degrees(self.alp[2])
			print "Gamma Config:"
			print "     W1:", np.degrees(self.gamma[0])," W2:", np.degrees(self.gamma[1])," W3:", np.degrees(self.gamma[2])
			print "Wheel Distance Config:"
			print "     l1:", self.l[0]," l2:", self.l[1]," l3:", self.l[2]
		print "Wheel Radius  : ", self.r
		print "Max RPM	       : ", self.max_rpm
		print "Min RPM	       : ", self.min_rpm
		print "Cut Off RPM   : ", self.cut_off_rpm
		print "cmd_vel topic : ", self.cmd_vel_topic
		print "motor topic   : ", self.motor_topic
		print "============================================="
		print "Motor Control is Ready..."
		self.delay.sleep()
		
	def get_invers_jacobian3(self):
		Ji = np.matrix([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]) #membuat matriks kosongan
		Ji[0,0] = np.cos(self.alp[0])
		Ji[0,1] = np.sin(self.alp[0])
		Ji[0,2] = self.l[0]*((np.cos(self.gamma[0])*np.sin(self.alp[0])) - (np.sin(self.gamma[0])*np.cos(self.alp[0])))
		Ji[1,0] = np.cos(self.alp[1])
		Ji[1,1] = np.sin(self.alp[1])
		Ji[1,2] = self.l[1]*((np.cos(self.gamma[1])*np.sin(self.alp[1])) - (np.sin(self.gamma[1])*np.cos(self.alp[1])))
		Ji[2,0] = np.cos(self.alp[2])
		Ji[2,1] = np.sin(self.alp[2])
		Ji[2,2] = self.l[2]*((np.cos(self.gamma[2])*np.sin(self.alp[2])) - (np.sin(self.gamma[2])*np.cos(self.alp[2])))
		Ji[3,0] = 0.
		Ji[3,1] = 0.
		Ji[3,2] = 0.
		return Ji*1/self.r
	
	def get_invers_jacobian4(self):
		Ji = np.matrix([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]) #membuat matriks kosongan
		Ji[0,0] = np.cos(self.alp[0])
		Ji[0,1] = np.sin(self.alp[0])
		Ji[0,2] = self.l[0]*((np.cos(self.gamma[0])*np.sin(self.alp[0])) - (np.sin(self.gamma[0])*np.cos(self.alp[0])))
		Ji[1,0] = np.cos(self.alp[1])
		Ji[1,1] = np.sin(self.alp[1])
		Ji[1,2] = self.l[1]*((np.cos(self.gamma[1])*np.sin(self.alp[1])) - (np.sin(self.gamma[1])*np.cos(self.alp[1])))
		Ji[2,0] = np.cos(self.alp[2])
		Ji[2,1] = np.sin(self.alp[2])
		Ji[2,2] = self.l[2]*((np.cos(self.gamma[2])*np.sin(self.alp[2])) - (np.sin(self.gamma[2])*np.cos(self.alp[2])))
		Ji[3,0] = np.cos(self.alp[3])
		Ji[3,1] = np.sin(self.alp[3])
		Ji[3,2] = self.l[3]*((np.cos(self.gamma[3])*np.sin(self.alp[3])) - (np.sin(self.gamma[3])*np.cos(self.alp[3])))		
		return Ji*1/self.r
		
	def cmd_vel_callback(self, dat):
		vx = dat.linear.x
		vy = dat.linear.y
		vz = dat.angular.z
		self.cmd_vel = np.matrix([[vx, vy, vz]]).T
		self.command_velocity_kinematics()
		
	def pwm_leveling(self, w):
		if(self.num_of_act == 4):
			temp = np.array([abs(w[0,0]), abs(w[1,0]), abs(w[2,0]), abs(w[3,0])])
		else:
			temp = np.array([abs(w[0,0]), abs(w[1,0]), abs(w[2,0], 0.)])
		out = np.matrix([[0., 0., 0., 0.]]).T			
		maximum = np.max(temp)
		if(maximum > self.max_rpm):
			out[0,0] = (w[0,0]/maximum)*self.max_rpm
			out[1,0] = (w[1,0]/maximum)*self.max_rpm
			out[2,0] = (w[2,0]/maximum)*self.max_rpm
			out[3,0] = (w[3,0]/maximum)*self.max_rpm
		else:
			out[0,0] = w[0,0]
			out[1,0] = w[1,0]
			out[2,0] = w[2,0]
			out[3,0] = w[3,0]
		#====================== W1 ====================
		if(out[0,0]>self.cut_off_rpm) and (out[0,0]<self.min_rpm):
			out[0,0] = self.min_rpm
		elif (out[0,0]<-self.cut_off_rpm) and (out[0,0]>-self.min_rpm):
			out[0,0] = -self.min_rpm
		#====================== W2 ====================
		if(out[1,0]>self.cut_off_rpm) and (out[1,0]<self.min_rpm):
			out[1,0] = self.min_rpm
		elif (w[1,0]<-self.cut_off_rpm) and (out[1,0]>-self.min_rpm):
			out[1,0] = -self.min_rpm
		#====================== W3 ====================
		if(out[2,0]>self.cut_off_rpm) and (out[2,0]<self.min_rpm):
			out[2,0] = self.min_rpm
		elif (out[2,0]<-self.cut_off_rpm) and (out[2,0]>-self.min_rpm):
			out[2,0] = -self.min_rpm
		#====================== W4 ====================
		if(out[3,0]>self.cut_off_rpm) and (out[3,0]<self.min_rpm):
			out[3,0] = self.min_rpm
		elif (out[3,0]<-self.cut_off_rpm) and (out[3,0]>-self.min_rpm):
			out[3,0] = -self.min_rpm
		#==============================================
		motor.x = out[0,0] #w1
		motor.y = out[1,0] #w2
		motor.z = out[2,0] #w3	
		motor.w = out[3,0] #w4		
		self.pwm_publisher.publish(motor)
		if(self.screen_output == True):
			print motor

	def command_velocity_kinematics(self): #manual mode
		w = self.Jr*self.cmd_vel
		self.pwm_leveling(w)

if __name__ == "__main__":
	rospy.init_node("robot1_motor_node")
	server = Server()
	motor = Quaternion()
	try:
		rospy.Subscriber(server.cmd_vel_topic, Twist, server.cmd_vel_callback)
		rospy.spin()
	except rospy.ROSInterruptException:
		pass
