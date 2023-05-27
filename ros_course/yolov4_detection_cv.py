import cv2
import numpy as np
import time

class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
net = cv2.dnn.readNet("yolov4-tiny.cfg", "yolov4-tiny.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
outNames = net.getUnconnectedOutLayersNames() 

list_images = ["image_0.jpg", "000008.png", "image_1.jpg", "image_2.jpg", "image_3.jpg", "image_4.jpg", "image_5.jpg"]
for img in list_images:
	start = time.time()
	frame = cv2.imread(img)
	blob = cv2.dnn.blobFromImage(frame, 1/255, size=(416, 416))
	net.setInput(blob)
	outs = net.forward(outNames)
	frameHeight = frame.shape[0]
	frameWidth = frame.shape[1]

	confThreshold = 0.5

	def drawPred(classId, conf, left, top, right, bottom):
		cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))
		label = '%.2f' % conf
		if class_names:
			assert(classId < len(class_names))
			label = '%s: %s' % (class_names[classId], label)
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
	print(time.time() - start)
	cv2.imwrite(img+"result_cv.jpg", frame) 
