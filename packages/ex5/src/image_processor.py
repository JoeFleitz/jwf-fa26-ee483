#!/usr/bin/env python3
"""
Exercise 5 image processing node.

One callback does everything:
  1. Convert the incoming ROS image to OpenCV (bgr8)
  2. Crop off the top half and publish on image_cropped
  3. Manual HSV white filter -> image_white
  4. Logistic regression white filter (ColorClassifier) -> image_white_ml
"""
import sys
from pathlib import Path

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# ml_filter.py and model_weights.npz both live in this folder (ex5/src).
# Adding it to the path makes the import work no matter how the node is started.
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from ml_filter import ColorClassifier  # noqa: E402


class ImageProcessor:
    def __init__(self):
        self.bridge = CvBridge()

        # Cropping. 0.5 keeps the bottom half of the image, as the assignment asks.
        self.crop_fraction = rospy.get_param("~crop_fraction", 0.5)

        # Manual HSV filter. White = low saturation, high value. Hue does not matter.
        self.hsv_lower = tuple(rospy.get_param("~hsv_lower", [0, 0, 170]))
        self.hsv_upper = tuple(rospy.get_param("~hsv_upper", [180, 60, 255]))
        self.erode_size = rospy.get_param("~erode_size", 3)
        self.dilate_size = rospy.get_param("~dilate_size", 5)

        # Logistic regression filter. The model is loaded ONCE here, never in the callback.
        param_path = str(SRC_DIR / "model_weights.npz")
        self.model = ColorClassifier(param_path)
        self.min_confidence_white = rospy.get_param("~min_confidence_white", 0.7)
        self.morph_kernel = rospy.get_param("~morph_kernel", 3)

        self.pub_cropped = rospy.Publisher("image_cropped", Image, queue_size=10)
        self.pub_white = rospy.Publisher("image_white", Image, queue_size=10)
        self.pub_white_ml = rospy.Publisher("image_white_ml", Image, queue_size=10)

        # Nodes start at different times, so give publishers a moment to connect
        rospy.sleep(1.0)
        rospy.Subscriber("image", Image, self.callback, queue_size=1)
        rospy.loginfo("ex5 image_processor ready")

    @staticmethod
    def _kernel(size):
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    def callback(self, msg):
        # ROS Image -> OpenCV BGR image
        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Crop: drop the top part, keep only the bottom (where the lanes are)
        height = cv_img.shape[0]
        start_row = int(height * self.crop_fraction)
        cropped = cv_img[start_row:, :]
        self.pub_cropped.publish(self.bridge.cv2_to_imgmsg(cropped, "bgr8"))

        # Filter 1: manual HSV white filter
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        if self.erode_size > 0:
            white_mask = cv2.erode(white_mask, self._kernel(self.erode_size))
        if self.dilate_size > 0:
            white_mask = cv2.dilate(white_mask, self._kernel(self.dilate_size))
        white_out = cv2.bitwise_and(cropped, cropped, mask=white_mask)
        self.pub_white.publish(self.bridge.cv2_to_imgmsg(white_out, "bgr8"))

        # Filter 2: logistic regression white filter (keep resize_to as given)
        masks = self.model.segment(
            cropped,
            min_confidence=self.min_confidence_white,
            morph_kernel=self.morph_kernel,
            resize_to=(160, 120),
        )
        white_ml = masks["White"].astype(np.uint8)
        # Safety check so bitwise_and never fails on a size mismatch
        if white_ml.shape[:2] != cropped.shape[:2]:
            white_ml = cv2.resize(
                white_ml, (cropped.shape[1], cropped.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        white_ml_out = cv2.bitwise_and(cropped, cropped, mask=white_ml)
        self.pub_white_ml.publish(self.bridge.cv2_to_imgmsg(white_ml_out, "bgr8"))


if __name__ == "__main__":
    rospy.init_node("image_processor", anonymous=False)
    ImageProcessor()
    rospy.spin()
