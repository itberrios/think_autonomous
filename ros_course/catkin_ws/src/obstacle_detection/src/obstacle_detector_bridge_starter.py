#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
# import CV BRIDGE and CV2

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
            RGB IMAGE
            # CONVERT TO GRAYSCALE
            ANY OPEN CV FUNCTION
            # CONVERT BACK TO ROS & PUBLISH

        except:
            CvBridgeError as e:
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