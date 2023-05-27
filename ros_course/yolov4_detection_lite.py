import cv2
import numpy as np
import tensorflow as tf
import time
from yolov4.tflite import YOLOv4

yolo = YOLOv4(tiny=True)

#yolo.classes = "coco.names"
#yolo.make_model()
#yolo.load_weights("yolov4-tiny.weights", weights_type="yolo")
yolo.load_tflite("yolov4.tflite")

list_images = ["image_0.jpg", "image_1.jpg", "image_2.jpg","image_3.jpg", "image_4.jpg", "image_5.jpg" ]

for img in list_images:
	start_time = time.time()

	original_image = cv2.imread(img)
	w, h, _ = original_image.shape
	resized_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
	#candidates = yolo.model.predict(input_data)
	pred_bboxes = yolo.predict(resized_image)
	pred_bb = []
	for b in pred_bboxes:
		print(b)
		b = [b[0]*w - b[2]*w/2, b[1]*h - b[3]*h/2, b[0]*w + b[2]*w/2, b[1]*h + b[3]*h/2, b[4], b[5]]
		pred_bb.append(b)
	print(pred_bb)
	exec_time = time.time() - start_time
	print("time: {:.2f} ms".format(exec_time * 1000))
	#result = yolo.draw_bboxes(original_image, pred_bboxes)
	result = cv2.rectangle(resized_image, pred_bb, (255,255,255))
	cv2.imwrite(img+"result_lite.jpg", result) 
