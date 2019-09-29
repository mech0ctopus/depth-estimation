#!/usr/bin/env python

import rosbag
import numpy as np
# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2

# Instantiate CvBridge
bridge = CvBridge()

def image_callback(msg,filename):
	'''Save color image to file.'''
	print "Received an image!"
	try:
		# Convert your ROS Image message to OpenCV2
		cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
	except CvBridgeError, e:
		print(e)
	else:
		# Save your OpenCV2 image as a jpeg
		path="/media/craig/CRAIG'S USB/bag_example_pics/"
		cv2.imwrite(path+filename+'.jpeg', cv2_img)

def depth_callback(msg,filename):
	'''Save depth image to file.'''
	print "Received depth!"
	try:
		NewImg = bridge.imgmsg_to_cv2(msg, "passthrough")
		depth_array = np.array(NewImg, dtype=np.float32)
		cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)
		# Save your OpenCV2 image as a jpeg
		path="/media/craig/CRAIG'S USB/bag_example_pics/"
		cv2.imwrite(path+filename+'.jpeg', depth_array*255)
	except CvBridgeError as e:
		print(e)

def get_first_msg(bag,topic):
	'''Returns type and contents of first message for given
	bag & topic.'''
	msgs=bag.read_messages(topics=topic)
	first_msg=msgs.next().message
	msg_type=type(first_msg)
	msg_shape=np.shape(np.array([first_msg]))
	#image_callback(first_msg)
	return msg_type, msg_shape, first_msg 

def get_all_msgs(bag,topic,image_type='color'):
	'''Saves all images to disk from a given bag & topic.'''
	msgs=bag.read_messages(topics=topic)

	if image_type=='color':
		callback=image_callback
	elif image_type=='depth':
		callback=depth_callback

	for idx,msg in enumerate(msgs):
		msg=msgs.next().message
		filename='image_'+str(idx)
		callback(msg,filename)

if __name__ == '__main__':
	bag = rosbag.Bag("/media/craig/CRAIG'S USB/2016-10-25-11-41-21_example.bag")
	topics=[#'/Multisense/depth',]
			#'/Multisense/left/image_rect_color',] 
			#'/Logitech_webcam/image_raw/compressed',]
			'/velodyne_packets']

	#get_all_msgs(bag,color_topics,)

	msg_types=[]
	msg_shapes=[]
	msgs=[]
	for topic in topics:
		msg_type, msg_shape, msg = get_first_msg(bag,topic)
		msg_types.append(msg_type)
		msg_shapes.append(msg_shape)
		msgs.append(msg)
	print msg_types
	# print msgs[0].height, msgs[0].width
	print dir(msgs[0])
	print msgs[0].packets
	data=msgs[0].packets
	print dir(data)
	test_shape=np.array([msgs[0].packets]).shape
	print test_shape
	#print msgs[1].height, msgs[1].width