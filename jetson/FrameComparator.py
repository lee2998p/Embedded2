import time
from functools import reduce

import cv2
import numpy as np


class FrameComparator:
    def __init__(self, ref_frame, change_time, change_thresh, blob_size=.1, conversion_code=cv2.COLOR_BGR2GRAY):
        """
        Creates a FrameComparator object
        @param ref_frame:  Initial frame to base comparison off of
        @param change_time: Time to allow ref_frame to persist before grabbing a new one
        @param change_thresh: The threshold before deciding that a pixel has changed enough
        @param blob_size: How much of the frame a blob must take up before deciding that the frame's contents have moved
        @param conversion_code: The OpenCV code to convert from the frame's colorspace to greyscale, set to None if already greyscale
        """
        self.conversion_code = conversion_code
        self.change_time = change_time
        self.change_thresh = change_thresh
        self.blob_size = blob_size
        self.ref_time = None
        self.ref_frame = None
        self.frame_size = None
        self.dilate_kern = np.ones((5, 5), np.uint8)
        self.set_ref_frame(ref_frame)

    def set_ref_frame(self, cur_frame):
        """
        Updates the reference frame used by the frame comparator
        @param cur_frame:  The current frame, that will become the new reference frame
        """
        self.frame_size = reduce((lambda x, y: x * y), cur_frame.size)
        if self.conversion_code is not None:
            self.ref_frame = cv2.cvtColor(cur_frame, self.conversion_code)
            self.frame_size /= cur_frame.shape[-1]  # Remove the last dim to get the number of pixels in the image
        else:
            # Last dim should either be 1 or not exist for greyscale
            self.ref_frame = cur_frame
        self.ref_time = time.time()

    def check_change(self, new_img):
        """
        Compares the new_img against an internal reference frame to check if there has been a significant enough change
        Updates ref_frame if a predetermined amount of time has elapsed
        @param new_img: The image to compare to the reference frame and potentially update the reference frame with
        @return: True if the image has changed, False if not
        """
        work_img = new_img
        if self.conversion_code is not None:
            work_img = cv2.cvtColor(new_img, self.conversion_code)
        blur = cv2.GaussianBlur(work_img, (21, 21), 0)
        frameDelta = cv2.absdif(self.ref_frame, blur)
        thresh = cv2.threshold(frameDelta, self.change_thresh, 255, cv2.THRESH_BINARY)[-1]
        thresh = cv2.dilate(thresh, self.dilate_kern, iterations=2)

        contours, _ = cv2.findContours(thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

        change_status = False

        for contour in contours:
            if cv2.contourArea(contour) > self.frame_size * self.blob_size:
                change_status = True
                break

        if time.time() - self.ref_time > self.change_time:
            self.set_ref_frame(new_img)

        return change_status
