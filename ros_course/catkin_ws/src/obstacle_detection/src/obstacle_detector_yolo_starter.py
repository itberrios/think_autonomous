#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import time

class ObstacleDetector(object):
    def __init__(self):
        self.image = Image()
        self.setup_ros()
        self.setup_yolo()
        self.loop()

    def setup_yolo(self):
        pass
    
    def setup_ros(self):
        """
        Build the node, the subscriber, and the publisher
        """
        rospy.init_node('python_image_subscriber')
        rospy.Subscriber("/image_raw", Image, self.image_callback, queue_size =1)
        self.pub = rospy.Publisher('/image_raw_processed', Image, queue_size=1)

    def image_callback(self,msg):
        """
        Build the image callback
        """
        self.image = msg

    def action_loop(self):
        try:
            # CONVERT FROM ROS TO OPENCV
            bridge = CvBridge()
            cv2_image = bridge.imgmsg_to_cv2(self.image,desired_encoding = "rgb8")
            # RUN TINY YOLO
            start = time.time()
            yolo_image = self.get_bounding_boxes(cv2_image)
            print("FPS:", 1/(time.time() - start))
            # CONVERT BACK TO ROS & PUBLISH
            image_ros = bridge.cv2_to_imgmsg(np.asarray(yolo_image), encoding='rgb8')
            return image_ros
        except CvBridgeError as e:
            print(e)
            return self.image
    
    def get_bounding_boxes(self, frame):
        return frame
    
    def loop(self):
        """
        Define the code that runs.
        Publishes another topic
        """
        rate = rospy.Rate(15)

        while not rospy.is_shutdown():
            ros_image = self.action_loop()
            self.pub.publish(ros_image)
            rate.sleep()
  
if __name__ == '__main__':
    ObstacleDetector()
