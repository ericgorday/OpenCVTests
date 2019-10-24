import cv2
import numpy as py
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
'''
HSV -> Hue, Saturation, Value; 
Hue is the region of the color wheel with different degree regions representing different colors, example: 0 < red < 60 deg
Saturation is amount of gray in something from 0 to 100 percent
Value is a measure of brightness of 0 to 100 percent; 0 is black, 100 is white

It's easier to represent colors in hsv than in bgr/rgb so we're first converting the object to a hsv before creating a color-based histogram
'''

def getTrainData(labelsEncoded):
	features = []
	labels  = []
	tests = ["test1.png"]
	for label in labelsEncoded.keys():
		for test in tests:
			picture_path = "vrx_pictures/" + label + "/" + test  #Path will change on each for loop iteration
    		BGRImage = cv2.imread(picture_path) 
    		hsvImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2HSV) #Changes color space to hsv
    		hue_hist = cv2.calcHist([hsvImage], [0], None, [180], [0, 180]) #Creates hue histogram
    		maxHue  = cv2.minMaxLoc(hue_hist)
    		maxHueBin = maxHue[3][1] #Finds modal hue represented in image/histogram
    		mask3 = cv2.inRange(hsvImage, py.array([maxHueBin - 2,0,0]), py.array([maxHueBin + 2,256,256])) #Mask containing small range around modal hue
    		mask3 = cv2.bitwise_not(mask3) #Mask that blocks out modal hue from orignal image
    		#Original modal hue is blocked out to take out water (all images will be taken from vrx navigator sim where most present hue is a blue representing water mesh)
    		hue_hist2 = cv2.calcHist([hsvImage], [0], mask3, [180], [0, 180]) #Second histogram calculated from image with mask
    		maxHue  = cv2.minMaxLoc(hue_hist2)
    		maxHueBin = maxHue[3][1] #Finds the second modal hue in orignal image
    		features.append(maxHueBin) #Appends this hue as a feature
    		labels.append(labelsEncoded[label]) #Appends this image's indexed integer label
	results = [features, labels]
	return results		

				
def trainGNB(features,labels,test):
	classifier = GaussianNB() #Naive Bayes Classifier
	classifier.fit(features,labels) #Training on feature and label set
	print("test h value: %s", test)
	prediction = classifier.predict(py.array(test).reshape(-1,1))
	print(prediction)

def getHistogram(path):
	BGRImage = cv2.imread(path)
	hsvImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2HSV)
	hue_hist = cv2.calcHist([hsvImage], [0], None, [180], [0, 180])
	maxHue  = cv2.minMaxLoc(hue_hist)
	maxHueBin = maxHue[3][1]
	mask3 = cv2.inRange(hsvImage, py.array([maxHueBin - 2,0,0]), py.array([maxHueBin + 2,256,256]))
	mask3 = cv2.bitwise_not(mask3)
	hue_hist2 = cv2.calcHist([hsvImage], [0], mask3, [180], [0, 180])
	plt.plot(hue_hist, color = "b")
	plt.show()
	plt.plot(hue_hist2, color = "r") # plots the modified histogram with mask
	plt.show()
	cv2.waitKey(0)
	maxHue  = cv2.minMaxLoc(hue_hist2)
	maxHueBin = maxHue[3][1]
	return maxHueBin #Represents second-most present color in orignal image

if __name__ == "__main__":
	labelsEncoded  = {'yellow_totem': 0, 'black_totem': 1, 'blue_totem': 2, 'green_totem': 3, 'red_totem':4, 
        'polyform_a3':5, 'polyform_a5':6, 'polyform_a7':7, 'surmark46104':8, 'surmark950400':9, 'surmark950410':10}
	results = getTrainData(labelsEncoded)
	test = getHistogram("vrx_pictures/blue_totem/test1.png") #Hard-coded specific object
	trainGNB(py.array(results[0]).reshape(-1,1), results[1],  test)









'''
	BGRImage = cv2.imread("test_pictures/blue.jpg")
	#BGRImage = cv2.imread("vrx_pictures/green_totem.png")
	hsvImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2HSV)
	color = ('b', 'g', 'r')
	lower_a = py.array([0,0,0])
	upper_a = py.array([100, 256, 256])
	lower_b = py.array([135,0 ,0])
	upper_b = py.array([180, 256, 256])
	mask1 = cv2.inRange(hsvImage,lower_a, upper_a)
	mask2 = cv2.inRange(hsvImage, lower_b, upper_b)
	mask = cv2.bitwise_or(mask1, mask2) # mask that supposedly blocks out blue color range
	
	
	#modifiedImage = cv2.bitwise_and(hsvImage,hsvImage, mask=mask3)
	
	hue_hist = cv2.calcHist([hsvImage], [0], None, [180], [0, 180]) # calculates a histogram based on the hue channel of the hsv image [0]
	sat_hist = cv2.calcHist([hsvImage], [1], None, [256], [0, 256]) # calculates a histogram based on the saturation channel of the hsv image [1]
	val_hist = cv2.calcHist([hsvImage], [2], None, [256], [0, 256]) # calculates a histogram based on the value channel of the hsv image [02
	
	maxHue  = cv2.minMaxLoc(hue_hist)
	print(maxHue[1])
	maxHueBin = maxHue[3][1]
	print(maxHueBin)

	mask3 = cv2.inRange(hsvImage, py.array([maxHueBin - 2,0,0]), py.array([maxHueBin + 2,256,256]))
	mask3 = cv2.bitwise_not(mask3)

	hue_hist2 = cv2.calcHist([hsvImage], [0], mask3, [180], [0, 180])

	histograms = [hue_hist,hue_hist2,sat_hist,val_hist]
	# Find modes of histogram over an  empiraccly defined threshold, and use that bin range for a mask to block out colors
	# After that is blocked out,  claculate histogram again on hue to find mode and use bin range as a feature
	for channel in histograms: # loops through each of the channel histograms created for the hsv image
		plt.plot(channel, color = "b") # plots the histogram of the specified channel
		plt.show()
		cv2.waitKey(0)
	#cv2.imshow("Image with mask", modifiedImage)
	#cv2.waitKey(0)
	'''
