import matplotlib.pyplot as plt
import cv2
import numpy as np
# from pyradar.classifiers.isodata import isodata_classification
from isodataclassifier import isodata_classification

def equalize_histogram(img, histogram, cfs):
    """
    Equalize pixel values to [0:255].
    """
    total_pixels = img.size
    N, M = img.shape
    min_value = img.min()
    L = 256  # Number of levels of grey
    cfs_min = cfs.min()
    img_corrected = np.zeros_like(img)
    corrected_values = np.zeros_like(histogram)

    divisor = np.float32(total_pixels) - np.float32(cfs_min)

    if not divisor:  # this happens when the image has all the values equals
        divisor = 1.0

    factor = (np.float32(L) - 1.0) / divisor

    corrected_values = ((np.float32(cfs) -
                         np.float32(cfs_min)) * factor).round()

    img_copy = np.uint64(img - min_value)
    img_corrected = corrected_values[img_copy]

    return img_corrected



def equalization_using_histogram(img):

    # Create histogram, bin edges and cumulative distributed function
    max_value = img.max()
    min_value = img.min()

    assert min_value >= 0, \
        "ERROR: equalization_using_histogram() img have negative values!"

    start, stop, step = int(min_value), int(max_value + 2), 1

    histogram, bin_edge = np.histogram(img, xrange(start, stop, step))
    cfs = histogram.cumsum()  # cumulative frencuency table
    img_corrected = equalize_histogram(img, histogram, cfs)

    return img_corrected


params = {"K": 100, "I" : 1000, "P" : 10, "THETA_M" : 10, "THETA_S" : 0.01,"THETA_C" : 8, "THETA_O" : 0.02}

img = cv2.imread('dataset/original/before.jpg',0)
# kernel = np.ones((5,5),np.uint8)
plt.imshow(img)
plt.show()

kernel 	= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
# print('Before')
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=7)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=7)


# print('Operated Image')
#plt.imshow(img)
#plt.show()


# img = cv2.imread('dataset/after.jpg',0)
# # kernel = np.ones((5,5),np.uint8)
# kernel 	= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
# print('After')
# img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=7)
# img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=7)


# print("After Operated Image")
# imgplot = plt.imshow(img)
# plt.show()



# run Isodata
class_image = isodata_classification(img, parameters=params)
# plt.imshow(class_image);
# plt.show()

# # equalize class image to 0:255

class_image_eq = equalization_using_histogram(class_image)

# # save it
save_image(IMG_DEST_DIR, "image_eq", image_eq)

# print("Equalized image classified using histogram 1")
# imgplot = plt.imshow(class_image_eq)
# plt.show()

# # also save original image
# image_eq = equalization_using_histogram(image)
# # save it
# print("Equalized image classified using histogram 2")
# #imgplot = plt.imshow(image_eq)
# #plt.show()

