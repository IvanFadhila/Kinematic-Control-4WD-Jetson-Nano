#!/usr/bin/env python

import rospy
import tf
import numpy as np
import time
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3, PoseWithCovarianceStamped
from std_msgs.msg import Int32, Int32MultiArray, Empty
from kinematic_control.srv import ImuRef, ImuRefResponse, ResetOdom, ResetOdomResponse
#from imu_bno055.msg import MyImu

class Server:
	def __init__(self):
		print "new odometry node"
		#====== Membaca parameter ROS untuk menentukan spesifikasi mekanik dan topic ========
		self.node_name = rospy.get_name() # membaca nama node yang dijalankan		
		self.amcl_pose_topic = rospy.get_param(self.node_name+"/amcl_pose_topic","amcl_pose") # membaca nama topic amcl 
		self.encoder_topic = rospy.get_param(self.node_name+"/encoder_topic","/robot/encoder") # membaca nama topic encoder 
		self.imu_topic = rospy.get_param(self.node_name+"/imu_topic", "/imu/data") # membaca nama topic imu
		self.imu_euler_topic = rospy.get_param(self.node_name+"/imu_euler_topic", "/imu/euler_data") # membaca nama topic imu dalam euler
		self.srv_set_pose_name = rospy.get_param(self.node_name+"/service_pose", "/robot/setPose") # membaca nama service untuk set inisial pose odometry
		self.srv_set_imu_name = rospy.get_param(self.node_name+"/service_imu_set","/robot/setImuRef") # membaca nama service untuk set zero reference imu
		self.odom_topic = rospy.get_param(self.node_name+"/odom_topic", "odom") # membaca nama topic odom untuk di publish
		self.odom_frame_id = rospy.get_param(self.node_name+"/odom_frame_id", "odom") # membaca nama frame id untuk odom
		self.child_frame_id = rospy.get_param(self.node_name+"/child_frame_id", "base_footprint") # membaca nama child frame untuk odom
		self.amcl_init_pose_topic = rospy.get_param(self.node_name+"/amcl_init_topic","initialpose") # membaca nama topic untuk inisialisasi pose amcl
		self.pose_topic = rospy.get_param(self.node_name+"/pose_topic","pose") # membaca nama topic untuk publish pose
		self.enc_num = rospy.get_param(self.node_name+"/number_of_encoders", 2) # membaca jumlah encoder yang digunakan
		self.r = rospy.get_param(self.node_name+"/encoder_wheel_radius", 0.025) # membaca radius roda encoder yang digunakan
		self.imu_reference = rospy.get_param(self.node_name+"/imu_reference", True) # membaca mode odometry mix dengan data imu
		self.enc_vel_reference = rospy.get_param(self.node_name+"/encoder_velocity_reference", False) # membaca data encoder dalam RPM atau Tick
		self.ppr = rospy.get_param(self.node_name+"/encoder_ppr", 360.) # membaca jumlah ppr dari encoder yang digunakan
		self.scale = rospy.get_param(self.node_name+"/odometry_scale", [1., 1., 1.]) # menentukan skala pengali data odometry
		self.output_screen = rospy.get_param(self.node_name+"/output_screen", True)
		if(self.enc_num == 3):		
			self.alp = rospy.get_param(self.node_name+"/alpha", [150., 30., -90.])
			self.gamma = rospy.get_param(self.node_name+"/gamma", [60., -60., 180.])
			self.l = rospy.get_param(self.node_name+"/wheels_distance", [0.24, 0.24, 0.24])
			for i in range(len(self.alp)):
				self.alp[i] = np.radians(self.alp[i])
				self.gamma[i] = np.radians(self.gamma[i])			
			self.Jr = self.get_jacobianR_3()			
		elif(self.enc_num == 4):
			self.alp = rospy.get_param(self.node_name+"/alpha", [135., 45., -135., -45])
			self.gamma = rospy.get_param(self.node_name+"/gamma", [45., -45., 135., 135])
			self.l = rospy.get_param(self.node_name+"/wheels_distance", [0.24, 0.24, 0.24, 0.24])
			for i in range(len(self.alp)):
				self.alp[i] = np.radians(self.alp[i])
				self.gamma[i] = np.radians(self.gamma[i])			
			self.Jr = self.get_jacobianR_4()			
		else:
			self.alp = rospy.get_param(self.node_name+"/alpha", [-135., -45.])
			self.gamma = rospy.get_param(self.node_name+"/gamma", [135., -135])
			self.l = rospy.get_param(self.node_name+"/wheels_distance", [0.24, 0.24])
			for i in range(len(self.alp)):
				self.alp[i] = np.radians(self.alp[i])
				self.gamma[i] = np.radians(self.gamma[i])
			self.Jr = self.get_jacobianR_2()
		self.w = np.matrix([[0., 0., 0., 0.]]).T
		self.last_w = np.matrix([[0., 0., 0., 0.]]).T
		#==================================================================================
		self.pose_publisher = rospy.Publisher(self.pose_topic, Vector3, queue_size = 1)
		self.odometry_publisher = rospy.Publisher(self.odom_topic, Odometry, queue_size=50)
		#self.amcl_pose_publisher = rospy.Publisher(self.amcl_init_pose_topic, PoseWithCovarianceStamped, queue_size=10)		
		self.pose = np.matrix([[0.0, 0.0, 0.0]]).T
		self.pose_des = np.matrix([[0.0, 0.0, 0.0]]).T   
		self.imu = 0.	  #robot heading from imu data
		self.imu_dot = 0. #angular velocity z from imu
		self.imu_raw = 0. #imu raw data
		self.imu_ref = 0. #imu zero reference
		self.imu_orientation = 0. #for imu reference data input
		self.current_time = rospy.Time.now()	# odometry time stamp
		self.last_time_s = rospy.Time.from_sec(time.time()).to_sec() # time sampling
		self.current_time_s = rospy.Time.from_sec(time.time()).to_sec() # time sampling
		self.counter = 0 #counter for amcl data input
		self.update_time = 0
		self.delay = rospy.Rate(1)	
		self.config_print()
	
	def config_print(self):
		print "======================================================================="
		print "Configuration"
		print "======================================================================="
		print "Odometry Using : ", self.enc_num, " Encoders" 
		if(self.enc_num == 3):
			print "Alpha Config:"
			print "     W1:", np.degrees(self.alp[0])," W2:", np.degrees(self.alp[1])," W3:", np.degrees(self.alp[2])
			print "Gamma Config:"
			print "     W1:", np.degrees(self.gamma[0])," W2:", np.degrees(self.gamma[1])," W3:", np.degrees(self.gamma[2])
			print "Wheel Distance Config:"
			print "     l1:", self.l[0]," l2:", self.l[1]," l3:", self.l[2]
		elif(self.enc_num == 4):
			print "Alpha Config:"
			print "     W1:", np.degrees(self.alp[0])," W2:", np.degrees(self.alp[1])," W3:", np.degrees(self.alp[2])," W4:", np.degrees(self.alp[3])
			print "Gamma Config:"
			print "     W1:", np.degrees(self.gamma[0])," W2:", np.degrees(self.gamma[1])," W3:", np.degrees(self.gamma[2]) ," W4:", np.degrees(self.gamma[3])
			print "Wheel Distance Config:"
			print "     l1:", self.l[0]," l2:", self.l[1]," l3:", self.l[2]," l4:", self.l[3]
		else:
			print "Alpha Config:"
			print "     EncoderL:", np.degrees(self.alp[0])," EncoderR:", np.degrees(self.alp[1])
			print "Gamma Config:"
			print "     EncoderL:", np.degrees(self.gamma[0])," EncoderR:", np.degrees(self.gamma[1])
			print "Wheel Distance Config:"
			print "     l1:", self.l[0]," l2:", self.l[1]
		print "Wheel Radius : ", self.r
		print "Encoder PPR  : ", self.ppr
		print "Imu Reference: ", self.imu_reference
		if(self.enc_vel_reference == True):
			print "Encoder Data :  RPM"
		else:
			print "Encoder Data :  Ticks"
		print "Odom Scale   : ", self.scale
		print "Output Screen: ", self.output_screen
		print "======================================================================="		
		print ("odometry is ready...")
		self.delay.sleep()
	
	def get_jacobianR_2(self):
		j = np.matrix([[0., 0., 0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.]])
		j[0,0] = np.cos(self.alp[0])
		j[0,1] = np.sin(self.alp[0])
		j[0,2] = (self.l[0]*np.cos(self.gamma[0])*np.sin(self.alp[0])) - (self.l[0]*np.sin(self.gamma[0])*np.cos(self.alp[0]))
		j[1,0] = np.cos(self.alp[1])
		j[1,1] = np.sin(self.alp[1])
		j[1,2] = (self.l[1]*np.cos(self.gamma[1])*np.sin(self.alp[1])) - (self.l[1]*np.sin(self.gamma[1])*np.cos(self.alp[1]))
		j[2,0] = 0.
		j[2,1] = 0.
		j[2,2] = 0.
		j[3,0] = 0.
		j[3,1] = 0.
		j[3,2] = 0.
		return self.r*j
	
	def get_jacobianR_3(self):
		j = np.matrix([[0., 0., 0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.]])
		j[0,0] = np.cos(self.alp[0])
		j[0,1] = np.sin(self.alp[0])
		j[0,2] = (self.l[0]*np.cos(self.gamma[0])*np.sin(self.alp[0])) - (self.l[0]*np.sin(self.gamma[0])*np.cos(self.alp[0]))
		j[1,0] = np.cos(self.alp[1])
		j[1,1] = np.sin(self.alp[1])
		j[1,2] = (self.l[1]*np.cos(self.gamma[1])*np.sin(self.alp[1])) - (self.l[1]*np.sin(self.gamma[1])*np.cos(self.alp[1]))
		j[2,0] = np.cos(self.alp[2])
		j[2,1] = np.sin(self.alp[2])
		j[2,2] = (self.l[2]*np.cos(self.gamma[2])*np.sin(self.alp[2])) - (self.l[2]*np.sin(self.gamma[2])*np.cos(self.alp[2]))		
		j[3,0] = 0.
		j[3,1] = 0.
		j[3,2] = 0.
		return self.r*j
	
	def get_jacobianR_4(self):
		j = np.matrix([[0., 0., 0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.]])
		j[0,0] = np.cos(self.alp[0])
		j[0,1] = np.sin(self.alp[0])
		j[0,2] = (self.l[0]*np.cos(self.gamma[0])*np.sin(self.alp[0])) - (self.l[0]*np.sin(self.gamma[0])*np.cos(self.alp[0]))
		j[1,0] = np.cos(self.alp[1])
		j[1,1] = np.sin(self.alp[1])
		j[1,2] = (self.l[1]*np.cos(self.gamma[1])*np.sin(self.alp[1])) - (self.l[1]*np.sin(self.gamma[1])*np.cos(self.alp[1]))
		j[2,0] = np.cos(self.alp[2])
		j[2,1] = np.sin(self.alp[2])
		j[2,2] = (self.l[2]*np.cos(self.gamma[2])*np.sin(self.alp[2])) - (self.l[2]*np.sin(self.gamma[2])*np.cos(self.alp[2]))
		j[3,0] = np.cos(self.alp[3])
		j[3,1] = np.sin(self.alp[3])
		j[3,2] = (self.l[3]*np.cos(self.gamma[3])*np.sin(self.alp[3])) - (self.l[3]*np.sin(self.gamma[3])*np.cos(self.alp[3]))		
		return self.r*j
	
	def get_jacobianW(self, th, Jr):
		rotZ = np.matrix([[np.cos(th), np.sin(th), 0.], [-np.sin(th), np.cos(th), 0.], [0., 0., 1.]])
		J = Jr * rotZ
		return J
	
	def setImuRef_service_callback(self, req): # set imu zero reference
		print "request data = ", req.SetImuRef
		self.imu_ref = self.imu_orientation
		print "IMU REF Updated!!! value : ", self.imu_ref
		return ImuRefResponse(1)
	
	def reset_odometry_service_callback(self, req): # set odometry initial pose data
		self.pose[0,0] = req.setPose.x
		self.pose[1,0] = req.setPose.y
		self.pose[2,0] = req.setPose.z
		print "Pose Updated!!! value : ", self.pose
		self.update_time = 0
		return ResetOdomResponse(1)
	
	def imu_callback(self, dat): # subscribe imu sensor euler data
		self.imu_orientation = dat.z
		self.imu_raw = dat.z
		if(self.imu_raw>180.):
			self.imu_raw = self.imu_raw - 360.
		if(self.imu_ref > 180.):
			self.imu_ref = self.imu_ref - 360.
		self.imu_raw = -self.imu_raw + self.imu_ref
		if(self.imu_raw < -180.):
			self.imu_raw = self.imu_raw + 360.
		elif(self.imu_raw > 180.):
			self.imu_raw = self.imu_raw - 360.
		self.imu = np.radians(self.imu_raw)
		self.pose[2,0] = self.imu
		
	def imu_velocity_callback(self, dat): # subscribe imu angular velocity z data
		self.imu_dot = dat.angular_velocity.z
		
	def encoder_callback(self, dat): # encoder data callback
		t = rospy.Time.from_sec(time.time())
		self.current_time_s = t.to_sec()
		self.dt = self.current_time_s - self.last_time_s
		self.last_time_s = self.current_time_s
		if(self.enc_vel_reference == True): # if data input in RPM
			self.w[0,0] = dat.x * 60. / 2 * np.pi
			self.w[1,0] = dat.y * 60. / 2 * np.pi
			self.w[2,0] = dat.z * 60. / 2 * np.pi
			self.w[3,0] = dat.w * 60. / 2 * np.pi
		else: # if data input in tick
			self.w[0,0] = 2*np.pi*self.r*(dat.x-self.last_w[0,0])/self.ppr
			self.w[1,0] = 2*np.pi*self.r*(dat.y-self.last_w[1,0])/self.ppr
			self.w[2,0] = 2*np.pi*self.r*(dat.z-self.last_w[2,0])/self.ppr
			self.w[3,0] = 2*np.pi*self.r*(dat.w-self.last_w[3,0])/self.ppr
			self.last_w[0,0] = dat.x
			self.last_w[1,0] = dat.y
			self.last_w[2,0] = dat.z
			self.last_w[3,0] = dat.w
		self.compute_odometry()
	
	def amcl_pose_callback(self, dat): # subscribe particle filter localization pose data
		new_amcl = dat
		new_amcl.header.stamp = rospy.Time.now()		
		#update odometry pose data when complete_flag is set
		quaternion = (
			new_amcl.pose.pose.orientation.x,
			new_amcl.pose.pose.orientation.y,
			new_amcl.pose.pose.orientation.z,
			new_amcl.pose.pose.orientation.w)
		euler = tf.transformations.euler_from_quaternion(quaternion)
		amcl_th = euler[2]
		#update amcl orientation data if the different between odometry pose and amcl pose more than 60 degrees
		if(abs(amcl_th - self.pose[2,0])>(np.radians(15.))) or (self.counter > 100): # update AMCL orientation
			new_amcl.pose.pose.orientation.x = self.odom_quat[0]
			new_amcl.pose.pose.orientation.y = self.odom_quat[1]			
			new_amcl.pose.pose.orientation.z = self.odom_quat[2]
			new_amcl.pose.pose.orientation.w = self.odom_quat[3]
			#new_amcl.pose.covariance[0] = 0.1
			#new_amcl.pose.covariance[7] = 0.1
			#new_amcl.pose.covariance[35] = 0.023
			tmp = list(new_amcl.pose.covariance)
			new_amcl.pose.covariance = tuple(tmp)
			self.amcl_pose_publisher.publish(new_amcl)
			print ("Update AMCL Pose")
			self.counter = 0
		self.counter = self.counter + 1
	
	def fused_odom_callback(self,dat):
		pose_data = dat
		if(self.update_time >= 50):
			#self.pose[0,0] = dat.pose.pose.position.x
			#self.pose[1,0] = dat.pose.pose.position.y
			self.update_time = 0
		self.update_time += 1
	
	def compute_odometry(self):
		#time stamped
		self.current_time = rospy.Time.now()		
		#==========================
		J = self.get_jacobianW(self.pose[2,0], self.Jr)
		Ji = np.linalg.pinv(J)
		pose_dot = Ji * self.w				
		pose_dot[0,0] = pose_dot[0,0] * self.scale[0]
		pose_dot[1,0] = pose_dot[1,0] * self.scale[1]
		pose_dot[2,0] = pose_dot[2,0] * self.scale[2]
		self.pose[0,0] = self.pose[0,0] + pose_dot[0,0] * self.dt#
		self.pose[1,0] = self.pose[1,0] + pose_dot[1,0] * self.dt#
		if(self.imu_reference == False):
			self.pose[2,0] = self.pose[2,0] + pose_dot[2,0] * self.dt#
		else:
			self.pose[2,0] = self.imu
			pose_dot[2,0] = self.imu_dot		
		self.imu_dot = 0		
		#======================= standard odometry ros =============================
		vx = pose_dot[0,0]
		vy = pose_dot[1,0]
		vth = pose_dot[2,0]		
		x = self.pose[0,0]
		y = self.pose[1,0]
		orientation = self.pose[2,0]
		#quartenion
		self.odom_quat = tf.transformations.quaternion_from_euler(0, 0, orientation)
		odom_broadcaster.sendTransform(
			(x, y, 0.),
			self.odom_quat,
			self.current_time,
			self.child_frame_id,
			self.odom_frame_id
		)
		odom.header.stamp = self.current_time
		odom.header.frame_id = self.odom_frame_id
		odom.pose.pose = Pose(Point(x, y, 0.), Quaternion(*self.odom_quat))
		#odom.pose.covariance[0] = 0.0001
		#odom.pose.covariance[7] = 0.0001
		#odom.pose.covariance[35] = 0.0001
		odom.child_frame_id = self.child_frame_id
		odom.twist.twist = Twist(Vector3(vx, vy, 0), Vector3(0, 0, vth))
		#odom.twist.covariance[0] = 0.00001
		#odom.twist.covariance[7] = 0.00001
		#odom.twist.covariance[35] = 0.00001
		#=======================================================================
		pose_data.x = self.pose[0,0]
		pose_data.y = self.pose[1,0]
		pose_data.z = self.pose[2,0]
		self.odometry_publisher.publish(odom)
		self.pose_publisher.publish(pose_data)
		if(self.output_screen == True):
			print self.pose

if __name__ == "__main__":
	rospy.init_node("robot_odometry_node")
	pose_data = Vector3()
	server = Server()
	odom = Odometry()
	new_amcl = PoseWithCovarianceStamped()
	odom_broadcaster = tf.TransformBroadcaster()
	try:
		rospy.Subscriber(server.encoder_topic, Quaternion, server.encoder_callback)
		rospy.Subscriber(server.imu_euler_topic, Vector3, server.imu_callback)
		rospy.Subscriber(server.imu_topic, Imu, server.imu_velocity_callback)
		rospy.Subscriber(server.amcl_pose_topic, PoseWithCovarianceStamped, server.amcl_pose_callback)
		#rospy.Subscriber("/robot1/fused_odom", Odometry, server.fused_odom_callback)
		rospy.Service(server.srv_set_imu_name, ImuRef, server.setImuRef_service_callback)
		rospy.Service(server.srv_set_pose_name, ResetOdom, server.reset_odometry_service_callback)
		rospy.spin()
	except rospy.ROSInterruptException:
		pass
