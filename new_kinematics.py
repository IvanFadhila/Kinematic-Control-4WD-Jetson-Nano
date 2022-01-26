#!/usr/bin/env python

import rospy
import numpy as np
import random
import tf
from numpy.linalg import inv
from geometry_msgs.msg import Vector3, Twist, Quaternion, PoseWithCovarianceStamped
from std_msgs.msg import Int32MultiArray, Float32MultiArray, Int32, Empty
from nav_msgs.msg import Odometry
from robsonema_service.srv import *
from robot_localization.srv import *

class Server:
	def __init__(self):
		#==================================================================================================
		#variabel kinematik
		#self.lamda = np.matrix([[50., 0., 0.], [0., 50., 0.], [0., 0., 50.]])		
		#self.lamda1 = np.matrix([[25., 0., 0.], [0., 25., 0.], [0., 0., 25.]])		
		#self.lamda2 = np.matrix([[22., 0., 0.], [0., 22., 0.], [0., 0., 22.]])
		self.lamda = np.matrix([[10., 0., 0.], [0., 10., 0.], [0., 0., 10.]])		
		self.lamda1 = np.matrix([[10., 0., 0.], [0., 10., 0.], [0., 0., 10.]])		
		self.lamda2 = np.matrix([[10., 0., 0.], [0., 10., 0.], [0., 0., 10.]])
		self.kp = np.matrix([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
		self.visual_lamda = np.matrix([[15., 0., 0.], [0., 15., 0.], [0., 0., 15.]])
		self.visual_lamda1 = np.matrix([[15., 0., 0.], [0., 15., 0.], [0., 0., 15.]])			
		#lambda untuk mengambil bola mati
		self.visual_lamda2 = np.matrix([[10., 0., 0.], [0., 10., 0.], [0., 0., 10.]])# > 2 m
		self.visual_lamda3 = np.matrix([[10., 0., 0.], [0., 10., 0.], [0., 0., 10.]])# < 2.0 m
		self.visual_lamda4 = np.matrix([[15., 0., 0.], [0., 15., 0.], [0., 0., 15.]])# < 1.5 m
		self.visual_lamda5 = np.matrix([[20., 0., 0.], [0., 20., 0.], [0., 0., 20.]])# < 1 m
		self.visual_getter_lamda = np.matrix([[25., 0., 0.], [0., 25., 0.], [0., 0., 25.]])
		#heading control gain
		#==================================================================================
		self.cmd_vel = np.matrix([[0., 0., 0.]]).T
		self.pose = np.matrix([[0., 0., 0.]]).T
		self.pose_des = np.matrix([[0., 0., 0.]]).T		
		self.last_pose_des = np.matrix([[0., 0., 0.]]).T
		self.ball_pos = np.matrix([[0., 0., 0.]]).T
		self.last_ball_pos = np.matrix([[0., 0., 0.]]).T
		self.ball_des = np.matrix([[0.15, 0., 0.]]).T	#desired saat mendekati bola permainan
		self.ball_des1 = np.matrix([[0., 0., 0.]]).T	#desired saat mendekati bola mati
		#=======================================================================================
		self.game_mode = 0.
		self.attack = 0.		
		self.goal_reached = 0.				#flag positioning
		self.lost_ball_secs = 0				#visual invers kinematics		
		self.lost_ball_thresh = 15			#data kehilangan bola sebelum stop
		#obstacle avoidance variables
		self.k_att = 0.5										#attractive force gain
		self.k_rep = 0.3
		self.k_lm = 0.825#0.875											#local minimal angle gain
		self.safe_lm = 0.5											#local minimal threshold
		self.safe_d = 0.85											#safe distance threshold between robot and obstacle
		self.detik = 0
		self.local_minima_d = 0.5						#local_minima_distance threshold between robot and obstacle
		self.v_attr = None
		self.v_reps = None
		self.obstacles = np.array([[100., 100.], [100., 100.]])
		#pengaman pwm
		self.num_lock = 0
		self.data_count = 0
		self.last_data_count = 0
		self.force_stop = 0
		self.stop_counter = 0
		#game variables
		self.flag_status = Int32MultiArray()
		self.flag_status.data = [0, 0]		
		self.position_code = 0		
		self.strategy = 1
		self.game_step = 0
		self.pass_ball = 0		
		self.robot1_ball_flag = 0
		#======================================
		#visual invers kinematics variables
		self.ball_error = 0.		
		self.lock = 0
		self.dribler_lock = 0
		self.stop_ball = 0		
		#======================================
		#heading control variables
		self.pass_complete = 0
		self.treshold_error = 0.4 									#threshold for position error
		self.keeper_pos = 2
		self.robot_arrived = 0
		self.counter = 0
		self.counter1 = 0
		print ("robot1 kinematic is ready!!!!")
		self.acceleration_time = 0
	
	def get_robot_perspective(self,th, x):
		rotZ = np.matrix([[np.cos(th), np.sin(th), 0.], [-np.sin(th), np.cos(th), 0.], [0., 0., 1]])
		return rotZ*x
	
	def publish_control(self, U):
		Ui = self.get_robot_perspective(self.pose[2,0],U)
		cmd_vel.linear.x = Ui[0,0]
		cmd_vel.linear.y = Ui[1,0]
		cmd_vel.linear.z = 0.
		cmd_vel.angular.x = 0.
		cmd_vel.angular.y = 0.
		cmd_vel.angular.z = Ui[2,0]
		cmd_vel_publisher.publish(cmd_vel)
		
	
	def attr_force(self, current_pos, goal, gain): # compute attractive force for potential field algorithm
		att = np.linalg.norm(goal - current_pos) * gain
		theta = np.arctan2((goal[1] - current_pos[1]),(goal[0] - current_pos[0]))
		return att, theta
	
	def reps_force(self, cur_pos, obs, gain, dst): # compute repulsive force for potential field algorithm
		reps = []
		thetas = []		
		current_pos = np.array([cur_pos[0,0], cur_pos[1,0]])
		for i in range(len(obs)):			
			Dobs = np.linalg.norm(obs[i] - current_pos.T)	
			if Dobs < dst :
				rep = 0.
				theta = 0.
				rep = gain * (1/Dobs - 1/dst) * 1/(Dobs**2)
				theta = np.arctan2((obs[i,1] - current_pos[1]),(obs[i,0] - current_pos[0]))
				reps.append(rep)				
				if(theta > np.pi):
					theta = (-np.pi) + (theta % np.pi)
				elif(theta < -np.pi):
					theta = (np.pi) + (theta % -np.pi)
				thetas.append(theta)
		return reps,thetas
	
	def game_mode_callback(self, dat): #callback untuk mode permainan (start, stop, positioning) dan status menyerang(menyerang/bertahan)
		self.game_mode = dat.data[0] # game mode: 1. Ball Charge, 2. Positioning
		if(dat.data[1] != self.position_code):
			self.acceleration_time = 0		
		self.position_code = dat.data[1]
		self.attack = dat.data[2]
		self.pass_complete = 0
		self.main()
	
	def booster_callback(self, dat):		
		self.main()
		
	def pose_callback(self, dat): #subscribe robot's pose in fused odom data
		robot_pose = dat
		#print dat
		quaternion = (
			robot_pose.pose.pose.orientation.x,
			robot_pose.pose.pose.orientation.y,
			robot_pose.pose.pose.orientation.z,
			robot_pose.pose.pose.orientation.w)
		euler = tf.transformations.euler_from_quaternion(quaternion)				
		self.pose[0,0] = dat.pose.pose.position.x
		self.pose[1,0] = dat.pose.pose.position.y
		self.pose[2,0] = euler[2]
		#print self.pose
		self.data_count += 1
		if(self.data_count>80):
			self.data_count = 0
		self.main()
	
	def pose1_callback(self, dat): #subscribe robot's pose in euler data
		self.pose[0,0] = dat.x
		self.pose[1,0] = dat.y
		self.pose[2,0] = dat.z
		print self.pose
	
	def pose_des_callback(self, dat): #subscribe desired pose for robot's movement
		self.pose_des[0,0] = dat.x
		self.pose_des[1,0] = dat.y
		self.pose_des[2,0] = dat.z
		er = self.pose_des - self.pose
		bearing = np.arctan2(er[1,0],er[0,0])
		if(self.pose_des[0,0] != self.last_pose_des[0,0]) or (self.pose_des[1,0] != self.last_pose_des[1,0]) or (self.pose_des[2,0] != self.last_pose_des[2,0]):
			self.robot_arrived = 0
			self.last_pose_des[0,0] = self.pose_des[0,0]
			self.last_pose_des[1,0] = self.pose_des[1,0]
			self.last_pose_des[2,0] = self.pose_des[2,0]
			self.acceleration_time = 0
	
	def ball_pos_callback(self, dat): #subscribe ball position from omni camera
		self.ball_pos[0,0] = dat.x
		self.ball_pos[1,0] = dat.y
		self.ball_pos[2,0] = dat.z
		#self.main()
	
	def ball_flag_callback(self, dat):
		print dat.z
				
	def obstacle_avoidance_kinematic(self): # potential field obstacle avoidance kinematics
		ers = np.matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 0.]])*(self.pose_des - self.pose)
		rep = 0.
		target_pose = self.pose_des			
		error = target_pose - self.pose
		#==================================
		#compute force
		v_attr, theta_attr = self.attr_force(self.pose, target_pose, self.k_att) # get F_attractive
		v_reps, theta_reps = self.reps_force(self.pose, self.obstacles, rep, self.safe_d) # get F_repulsive and F_lm
		x_dot = v_attr * np.cos(theta_attr)
		y_dot = v_attr * np.sin(theta_attr)
		obs_sum = len(v_reps)	
		for j in range(len(v_reps)): # adding x_dot and y_dot with F_repulsive and F_lm
			x_dot = x_dot - v_reps[j] * np.cos(theta_reps[j])
			y_dot = y_dot - v_reps[j] * np.sin(theta_reps[j])		
			lm_rate = v_attr/v_reps[j]
			if(lm_rate < self.safe_lm): # local minimal addition
				if(theta_attr>=np.radians(0.))and(theta_attr<=np.radians(90.)):
					if(theta_reps[j]>np.radians(90.))and(theta_reps[j]<=np.radians(180.)):
						ang = np.pi
					else:
						ang = np.pi
				elif(theta_attr>np.radians(90.))and(theta_attr<=np.radians(180.)):
					if(theta_reps[j]>=np.radians(0.))and(theta_reps[j]<=np.radians(90.)):
						ang = np.pi
					else:
						ang = np.pi
				elif(theta_attr<np.radians(0.))and(theta_attr>=np.radians(-90.)):
					if(theta_reps[j]>=np.radians(0.))and(theta_reps[j]<=np.radians(180.)):
						ang = np.pi
					else:
						ang = -np.pi
				elif(theta_attr<np.radians(-90.))and(theta_attr>=np.radians(-180.)):
					if(theta_reps[j]>=np.radians(0.))and(theta_reps[j]<=np.radians(180.)):
						ang = -np.pi
					else:
						ang = np.pi
				x_dot = x_dot - v_reps[j] * np.cos(theta_reps[j] + self.k_lm * ang) # adding x_dot with F_lm if F_lm available
				y_dot = y_dot - v_reps[j] * np.sin(theta_reps[j] + self.k_lm * ang) # adding y_dot with F_lm if F_lm available
		error[0,0] = x_dot
		error[1,0] = y_dot
		self.kp = self.lamda
		tresh = 0.4
		if(np.linalg.norm(ers) < tresh):
			error = error * 0
		dribler.x = 0
		dribler.y = 0
		self.acceleration_time += 1	
		if(self.acceleration_time < 10):
			gain = self.kp * 0.5
		elif(self.acceleration_time < 15):
			gain = self.kp * 0.75
		elif(self.acceleration_time < 25):
			gain = self.kp * 0.85
		else:
			gain = self.kp
		U = gain*error		
		dribler_publisher.publish(dribler)
		self.publish_control(U)
	
	def visual_invers_kinematic(self): # kinematik saat mengejar bola
		#digunakan ketika robot kehilangan bola
		#robot tidak langsung berhenti, tapi tetap bergerak ke arah dimana bola terdeteksi untuk beberapa saat
		if(np.linalg.norm(self.ball_pos) == 0):
			self.lost_ball_secs += 1
		else:
			self.ball_error = self.ball_pos - self.ball_des
			self.lost_ball_secs = 0
		if(self.lost_ball_secs > self.lost_ball_thresh):
			self.ball_error = self.ball_error * 0				
		#print self.ball_error
		if(np.linalg.norm(self.ball_error) != 0):
			if(self.ball_pos[2,0] > 0.6) or (self.ball_pos[2,0] < -0.6):
				self.lock = 0
			elif(self.ball_pos[0,0] < 0.75):
				self.lock = 1
			if(self.ball_pos[0,0] < 1.25)and(self.lock == 0):
				dribler.x = 175
				dribler.y = 175
			else:
				dribler.x = 0
				dribler.y = 0
		else:
			self.lock = 0
		if(self.robot1_ball_flag != 0):
			dribler.x = 180
			dribler.y = 180
			self.lock = 0
			self.ball_error = self.ball_error * 0
			gain = 0
		elif(e_norm < 1.0):
			gain = self.visual_lamda5
		elif(e_norm < 1.5):
			gain = self.visual_lamda4
		elif(e_norm < 2.0):
			gain = self.visual_lamda3	
		else:
			gain = self.visual_lamda2
			self.flag_status.data[1] = 0
		if(self.lock==1):			
			gain = self.visual_lamda3
			dribler.x = 180
			dribler.y = 180
			U = np.matrix([[0., 0., 0.]]).T
			U[0,0] = 1.0				
			if(self.ball_error[2,0] <= 0.1 and self.ball_error[2,0] >= -0.1):
				U[1,0] = 0.0
				U[2,0] = 0.0
			else:
				U[1,0] = 0.0
				U[2,0] = self.ball_error[2,0]				
		else:
			U = gain*self.ball_error
		dribler_publisher.publish(dribler)
		self.publish_control(U)		
	
	def reset_pose1_f(self): #reset posisi amcl
		new_amcl1 = PoseWithCovarianceStamped()
		new_amcl1.pose.pose.position.x = 1.5
		new_amcl1.pose.pose.position.y = 0.7
		new_amcl1.pose.pose.position.z = 0.
		new_amcl1.pose.pose.orientation.x = 0.
		new_amcl1.pose.pose.orientation.y = 0.
		new_amcl1.pose.pose.orientation.z = 0.0051
		new_amcl1.pose.pose.orientation.w = 0.99
		new_amcl1.pose.covariance[0] = 0.1
		new_amcl1.pose.covariance[7] = 0.1
		new_amcl1.pose.covariance[35] = 0.068
		amcl_pose_publisher1.publish(new_amcl1)
		reset_pose = rospy.ServiceProxy('/robot1/set_pose', SetPose)
		a = SetPoseRequest()
		a.pose.header.frame_id = 'map'
		a.pose.pose.pose.position.x = 1.5
		a.pose.pose.pose.position.y = 0.7
		a.pose.pose.pose.position.z = 0.0
		reset_pose(a)

	def stop_motor(self):
		cmd_vel.linear.x = 0.
		cmd_vel.linear.y = 0.
		cmd_vel.linear.z = 0.
		cmd_vel.angular.x = 0.
		cmd_vel.angular.y = 0.
		cmd_vel.angular.z = 0.
		dribler.x = 0
		dribler.y = 0
		cmd_vel_publisher.publish(cmd_vel)
		dribler_publisher.publish(dribler)

	def main(self):
		if(self.last_data_count != self.data_count):
			self.last_data_count = self.data_count
			self.force_stop = 0
			self.stop_counter = 0
		else:
			self.stop_counter += 1
			#if(self.stop_counter > 10):
			#	self.force_stop = 1				
		self.num_lock += 1;
		if(self.num_lock > 50):
			self.num_lock = 0
			
		if(self.game_mode == 0):#mode stop
			self.acceleration_time = 0
			self.stop_motor()
					
		elif(self.game_mode == 1): #mode mencari bola dan menendang ke gawang
			if(self.force_stop == 0):
				self.visual_invers_kinematic()
			else:
				self.stop_motor()					
		
		elif(self.game_mode == 2) or (self.game_mode == 3): #mode positioning dan check lapangan
			if(self.force_stop == 0):
				self.obstacle_avoidance_kinematic()				
			else:
				self.stop_motor()
		
		elif(self.game_mode == 4):#mode check control arah hadap terhadap robot 2
			if(self.force_stop == 0):
				self.heading_control_kinematics()
			else:
				self.stop_motor()
					
if __name__ == "__main__":
	rospy.init_node("robot1_control_node")
	cmd_vel_publisher = rospy.Publisher("robot1/cmd_vel", Twist, queue_size = 3)
	dribler_publisher = rospy.Publisher("robot1/dribler", Vector3, queue_size = 3)
	goal_reached_publisher = rospy.Publisher("robot1/command_flag", Int32MultiArray, queue_size = 1)
	amcl_pose_publisher1 = rospy.Publisher("/robot1/initialpose", PoseWithCovarianceStamped, queue_size=10)
	server = Server()
	dribler = Vector3()
	cmd_vel = Twist()
	try:
		rospy.Subscriber("/robot1/fused_odom", Odometry, server.pose_callback)
		#rospy.Subscriber('/robot1/amcl_pose', PoseWithCovarianceStamped, server.amcl_pose_callback)
		#rospy.Subscriber("/robot1/pose", Vector3, server.pose1_callback)
		rospy.Subscriber("/robot1/target_pose", Vector3, server.pose_des_callback)
		rospy.Subscriber("/robot1/ball_position", Vector3, server.ball_pos_callback)
		rospy.Subscriber("/robot1/booster", Empty, server.booster_callback)
		#rospy.Subscriber("/robot2/pose", Vector3, server.robot2_pose_callback)
		rospy.Subscriber("/team/base_station_command", Int32MultiArray, server.game_mode_callback)
		#rospy.Subscriber("/team/team_strategy", Float32MultiArray, server.team_strategy_callback)		
		rospy.spin()
	except rospy.ROSInterruptException:
		pass
