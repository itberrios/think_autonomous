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
        class_names = []
        with open("/home/think/catkin_ws/src/obstacle_detection/src/weights/classes.txt", "r") as f:
            self.class_names = [cname.strip() for cname in f.readlines()]
        self.net = cv2.dnn.readNet("/home/think/catkin_ws/src/obstacle_detection/src/weights/yolov4-tiny.cfg", "/home/think/catkin_ws/src/obstacle_detection/src/weights/yolov4-tiny.weights")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.outNames = self.net.getUnconnectedOutLayersNames() 
    
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
        blob = cv2.dnn.blobFromImage(frame, 1/255, size=(416, 416))
        self.net.setInput(blob)
        outs = self.net.forward(self.outNames)
        to_return = self.postprocess(frame, outs)
        return to_return
    
    def postprocess(self,frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        net = self.net
        confThreshold = 0.5

        def drawPred(classId, conf, left, top, right, bottom):
            # Draw a bounding box.
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

            label = '%.2f' % conf

            # Print a label of class.
            if self.class_names:
                assert(classId < len(self.class_names))
                label = '%s: %s' % (self.class_names[classId], label)

            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])
            cv2.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        layerNames = net.getLayerNames()
        lastLayerId = net.getLayerId(layerNames[-1])
        lastLayer = net.getLayer(lastLayerId)

        classIds = []
        confidences = []
        boxes = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
        indices = np.arange(0, len(classIds))

        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
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