#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image

class ObstacleDetector(object):
    def __init__(self):
        self.image = Image()
        self.setup_ros()
        self.loop()

    def setup_ros(self):
        """
        Build the node, the subscriber, and the publisher
        """
        pass

    def image_callback(self,msg):
        """
        Build the image callback
        """
        pass

    def loop(self):
        """
        Define the code that runs.
        Publishes another topic
        """
        rate = rospy.Rate(15)

        while not rospy.is_shutdown():
            #PUBLISH HERE
            rate.sleep()
  
if __name__ == '__main__':
    ObstacleDetector()