# USAGE
# python brightness-calc.py --dir directory

# import the necessary packages
import argparse
import imutils
import cv2
import glob
import numpy as np


def crop_minAreaRect(img, rect):
    # rotate img
    angle = rect[2]
    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rot = cv2.warpAffine(img, M, (cols, rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1],
               pts[1][0]:pts[2][0]]

    return img_crop


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", required=True,
                help="path to the folder containing the images")
args = vars(ap.parse_args())

# load the images
print(args["dir"] + '/*png')
images = [cv2.imread(file) for file in glob.glob(args["dir"] + '/*pgm')]

# to store the means
means = []

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

    # extract the cropped area
    image = crop_minAreaRect(image, compRect)

    # TODO register the output image in folder *dir*_cropped

    # show the output image
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)

    # get the mean value
    mean, sd = cv2.meanStdDev(image)
    means.append(float(mean[0]))

print(np.mean(means))
