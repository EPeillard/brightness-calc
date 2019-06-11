# USAGE
# python brightness-calc.py --dir directory

# import the necessary packages
import argparse
import imutils
import cv2
import glob
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", required=True,
                help="path to the folder containing the images")
args = vars(ap.parse_args())

# load the images
print(args["dir"] + '/*png')
images = [cv2.imread(file) for file in glob.glob(args["dir"] + '/*pgm')]

# loop over the images
for image in images:

    # resize it to a smaller factor so that
    # the shapes can be approximated better
    resized = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resized.shape[0])

    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY)[1]

    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours and find the biggest one
    maxArea = 0
    for c in cnts:
        # compute the size of the contour
        size = cv2.contourArea(c)

        if size > maxArea:
            maxArea = size
            rect = c

    # multiply the contour (x, y)-coordinates by the resize ratio
    rect = rect.astype("float")
    rect *= ratio
    rect = rect.astype("int")

    # then compute the bounding box
    peri = cv2.arcLength(rect, True)
    approx = cv2.approxPolyDP(rect, 0.04 * peri, True)
    compRect = cv2.minAreaRect(approx)

    # draw original contour
    cv2.drawContours(image, rect, -1, (0, 255, 0), 2)
    # and rectangle contour
    box = cv2.boxPoints(compRect)
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

    # show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
