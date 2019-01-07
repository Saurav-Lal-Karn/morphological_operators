import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np

for filename in glob.glob('dataset/*.jpg'): #assuming jpg
	img = cv2.imread(filename,0)
	# kernel = np.ones((5,5),np.uint8)
	kernel 	= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
	img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=7)
	img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=7)
	print(filename)
	imgplot = plt.imshow(img)
	plt.show()
