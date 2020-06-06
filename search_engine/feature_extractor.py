import numpy as np 
import cv2
import imutils


class ColorDescriptor:

    def __init__(self, bins):
        self.bins = bins

    def histogram(self, image, mask):
        #extract histogram for the mask region
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,
            [0, 180, 0, 256, 0, 256])

        # normalize the histogram if we are using OpenCV 2.4
        if imutils.is_cv2():
            hist = cv2.normalize(hist).flatten()
        else:
            hist = cv2.normalize(hist, hist).flatten()

        hist = [np.round(h, 3) for h in hist]
        return hist

    def describe(self, image):
        # convert image to hsv space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # initialize features
        features = []

        # height, width, compute centre dimensions
        (h, w) = image.shape[:2]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))

        # rectangle segments
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), \
            (cX, w, cY, h), (0, cX, cY, h)]

        # an elliptical mask
        (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
        ellipMask = np.zeros((h, w), dtype = "uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

        # loop over the segments
        for (start_x, end_x, start_y, end_y) in segments:

            # mask for each corner
            cornerMask = np.zeros((h, w), dtype = "uint8")
            cv2.rectangle(cornerMask, (start_x, start_y), (end_x, end_y), 255, -1)
            cornerMask = cv2.subtract(cornerMask, ellipMask)

            hist = self.histogram(image, cornerMask)
            features.extend(hist)

        hist = self.histogram(image, ellipMask)
        features.extend(hist)

        return features


