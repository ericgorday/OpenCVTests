import cv2
import numpy as py
from matplotlib import pyplot as plt
'''
HSV -> Hue, Saturation, Value; 
Hue is the region of the color wheel with different degree regions representing different colors, example: 0 < red < 60 deg
Saturation is amount of gray in something from 0 to 100 percent
Value is a measure of brightness of 0 to 100 percent; 0 is black, 100 is white

It's easier to represent colors in hsv than in bgr/rgb so we're first converting the object to a hsv before creating a color-based histogram
'''
if __name__ == "__main__":
	BGRImage = cv2.imread("test_pictures/rainbow.png")
	hsvImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2HSV)
	color = ('b', 'g', 'r')
	lower_a = py.array([0,0,0])
	upper_a = py.array([100, 256, 256])
	lower_b = py.array([135,0 ,0])
	upper_b = py.array([180, 256, 256])
	mask1 = cv2.inRange(hsvImage,lower_a, upper_a)
	mask2 = cv2.inRange(hsvImage, lower_b, upper_b)
	mask = cv2.bitwise_or(mask1, mask2) # mask that supposedly blocks out blue color range
	
	# I can also consider creating a 2D or 3D histogram to combine the three channels into one
	hue_hist = cv2.calcHist([hsvImage], [0], None, [180], [0, 180]) # calculates a histogram based on the hue channel of the hsv image [0]
	sat_hist = cv2.calcHist([hsvImage], [1], None, [256], [0, 256]) # calculates a histogram based on the saturation channel of the hsv image [1]
	val_hist = cv2.calcHist([hsvImage], [2], None, [256], [0, 256]) # calculates a histogram based on the value channel of the hsv image [02
	histograms = [hue_hist,sat_hist,val_hist]
	
	for channel in histograms: # loops through each of the channel histograms created for the hsv image
		plt.plot(channel, color = "b") # plots the histogram of the specified channel
		plt.show()
		cv2.waitKey(0)