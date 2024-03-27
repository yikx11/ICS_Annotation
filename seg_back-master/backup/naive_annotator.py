import cv2
import numpy as np
import PIL.Image as Image
import torch


class NaiveAnnotator():

    def __init__(self):
        pass

    def predict(self, image, bbox):
        xmin = int(bbox['xmin'] + 0.5)
        ymin = int(bbox['ymin'] + 0.5)
        xmax = int(bbox['xmin'] + bbox['width'])
        ymax = int(bbox['ymin'] + bbox['height'])
        crop_image = image[ymin : ymax, xmin : xmax]
        gray_image = cv2.cvtColor(crop_image, cv2.COLOR_RGB2GRAY)
        canny_image = cv2.Canny(gray_image, 50, 150)
        h = cv2.findContours(
            canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # contours.shape = (*, 1, 2)
        contours = h[0]
        # contours = [
        #     cv2.approxPolyDP(contour, 50, True) for contour in contours
        # ]
        contours.sort(key=lambda x : cv2.contourArea(x), reverse=True)
        contours = [cv2.convexHull(contour) for contour in contours]
        contour = contours[0].astype(np.float)
        res = [
            {'x': xmin + point[0, 0], 'y': ymin + point[0, 1]}
            for point in contour
        ]
        return res

    def update(self):
        pass