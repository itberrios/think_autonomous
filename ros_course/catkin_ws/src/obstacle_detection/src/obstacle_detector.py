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
        rospy.init_node('python_image_subscriber')
        rospy.Subscriber("/image_raw", Image, self.image_callback, queue_size =1)
        self.pub = rospy.Publisher('/image_raw_processed', Image, queue_size=1)

    def image_callback(self,msg):
        """
        Build the image callback
        """
        self.image = msg

    def loop(self):
        """
        Define the code that runs.
        Publishes another topic
        """
        rate = rospy.Rate(15)

        while not rospy.is_shutdown():
            self.pub.publish(self.image)
            rate.sleep()
  
if __name__ == '__main__':
    ObstacleDetector()