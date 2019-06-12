# USAGE
# python brightness-calc.py --dir directory

# import the necessary packages
import argparse
import imutils
import cv2
import glob
import numpy as np
import os


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
                help="path to the folder containing the images (can use several, comma separated)")
ap.add_argument('-v', action='store_true',
                help="verbose mode, display the images one by one")
ap.add_argument('-c', '--crop', default=0, type=float,
                help="crop the image to reduce each dimension by the chosen value")
args = vars(ap.parse_args())

dirs = args["dir"].split(",")

for dir in dirs:
    # load the images
    images = [(cv2.imread(file), os.path.split(os.path.splitext(file)[0])[1]) for file in glob.glob(dir + '/*pgm')]

    # to store the means
    means = []

    # register the output image in folder *dir*_cropped
    path = dir + "_cropped"

    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed (maybe it already exists)" % path)
    else:
        print("Successfully created the directory %s " % path)

    # loop over the images
    for image, name in images:

        # resize it to a smaller factor so that
        # the shapes can be approximated better
        # => removed
        resized = image # imutils.resize(image, width=300)
        ratio = image.shape[0] / float(resized.shape[0])

        # convert the resized image to grayscale, blur it slightly,
        # and threshold it
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)[1]

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

        # if required, crop the image more
        if args["crop"] > 1:
            (x, y), (width, height), angle = compRect
            compRect = ((x, y), (width/args["crop"], height/args["crop"]), angle)

        # extract the cropped area
        cropIm = crop_minAreaRect(image, compRect)

        # get the mean value
        mean, sd = cv2.meanStdDev(cropIm)
        means.append(float(mean[0]))

        # save the image
        new_extension = '.png'
        completePath = path+"\\"+name+new_extension
        cv2.imwrite(completePath, cropIm)

        # show the output image
        if args["v"]:
            cv2.drawContours(image, rect, -1, (0, 255, 0), 2)
            box = cv2.boxPoints(compRect)
            box = np.int0(box)
            cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
            cv2.imshow("Image", image)
            cv2.waitKey(0)

    print(np.mean(means))
