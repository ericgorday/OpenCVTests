import cv2
import numpy as py

if __name__ == '__main__': 
	# program displays a picture and allows the user to manually select the roi using a rectangle created by the mouse input
	image = cv2.imread("test_pictures/rainbow.png")
	fromCenter = False
	showCrossHair = False
	(x,y,w,h) = cv2.selectROI("Image", image, fromCenter, showCrossHair)
	print("Region of interest recieved")
	imageROI = image[y:y+h, x:x+w]
	cv2.imshow("ROI", imageROI)
	cv2.waitKey(0)
