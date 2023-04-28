# Python code for Multiple Color Detection
# Link to code: https://www.geeksforgeeks.org/multiple-color-detection-in-real-time-using-python-opencv/
# Authors: Lydia Yang, Petra Ilic, Katie Yurechko, John Adekola

import numpy as np
import cv2


# Capturing video through webcam
webcam = cv2.VideoCapture(0)
areas = []
# Start a while loop
while(1):
	
	# Reading the video from the
	# webcam in image frames
	_, imageFrame = webcam.read()

	# Convert the imageFrame in
	# BGR(RGB color space) to
	# HSV(hue-saturation-value)
	# color space
	hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)


	# Set range for blue color and
	# define mask
	blue_lower = np.array([94, 80, 2], np.uint8)
	blue_upper = np.array([120, 255, 255], np.uint8)
	blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

	# Morphological Transform, Dilation
	# for each color and bitwise_and operator
	# between imageFrame and mask determines
	# to detect only that particular color
	kernel = np.ones((5, 5), "uint8")


	# Creating contour to track blue color
	contours, hierarchy = cv2.findContours(blue_mask,
										cv2.RETR_TREE,
										cv2.CHAIN_APPROX_SIMPLE)
	
	

    # Sort the blobs by size, largest to smallest
	contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Get the biggest 3 blobs
	biggest_blobs = contours[:3]
	for pic, contour in enumerate(biggest_blobs):
		x, y, w, h = cv2.boundingRect(contour)
		imageFrame = cv2.rectangle(imageFrame, (x, y),
									(x + w, y + h),
									(255, 0, 0), 2)	
		cv2.putText(imageFrame, "Blue", (x, y),
						cv2.FONT_HERSHEY_SIMPLEX,
						1.0, (255, 0, 0))
	
			
	# Program Termination
	cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame)
	if cv2.waitKey(1) == 27:
		break
webcam.release()
cv2.destroyAllWindows()