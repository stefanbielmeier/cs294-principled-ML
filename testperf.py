import matplotlib as plt
import numpy as np

import rfrgb
import rfgrayscale


def main():

	accuracy = 0
	rgbset = []
	grayset = []


	with open('cifar10_grayscale_test.csv', 'r') as file:

		gray = np.genfromtxt(file, delimiter=',',dtype=None, encoding=None)
		graydata = gray[1:,:]
		grayset = graydata[:,:-1]
		graylabels = graydata[:,-1]

	rgbpredclasses = rfgrayscale.predict(grayset, remap=True)
	correctclassifications = rgbpredclasses == graylabels			
	numcorrect = np.size(correctclassifications[correctclassifications == True])
	numrows = grayset.shape[0]
	accuracy = numcorrect / numrows

	print("Test accuracy :", accuracy, " (", numcorrect, "/", numrows, ")")

	with open('cifar10_rgb_v1_test.csv', 'r') as file:

		rgb = np.genfromtxt(file, delimiter=',',dtype=None, encoding=None)
		rgbdata = rgb[1:,:]
		grayset = rgbdata[:,:-1]
		rgblabels = rgbdata[:,-1]

	rgbpredclasses = rfrgb.predict(grayset, remap=True)
	correctclassifications = rgbpredclasses == rgblabels			
	numcorrect = np.size(correctclassifications[correctclassifications == True])
	numrows = grayset.shape[0]
	accuracy = numcorrect / numrows

	print("Test accuracy :", accuracy, " (", numcorrect, "/", numrows, ")")

	

if __name__ == '__main__':
	main()