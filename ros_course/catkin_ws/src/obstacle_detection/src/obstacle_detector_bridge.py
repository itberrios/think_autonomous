#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
# import CV BRIDGE
from cv_bridge import CvBridge, CvBridgeError
import cv2

class ObstacleDetector(object):
    def __init__(self):
        self.image = Image()
        self.setup_ros()
        self.loop()

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
            # CONVERT TO GRAYSCALE
            cv2_grayscale = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2GRAY)
            # CONVERT BACK TO ROS & PUBLISH
            image_ros = bridge.cv2_to_imgmsg(cv2_grayscale)
            return image_ros
        except CvBridgeError as e:
            print(e)
            return self.image
            
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