#!/usr/bin/env python3
"""
Exercise 6, Node 1 (copy of the Ex5 node, HSV filter only).

  1. Convert the incoming ROS image to OpenCV (bgr8)
  2. Crop off the top half and publish on image_cropped
  3. Manual HSV white filter, publish the BINARY mask (mono8) on image_white

The LR filter was removed. Both outputs share the same header so the
lane lines node can sync them with message_filters.
"""
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ImageProcessor:
    def __init__(self):
        self.bridge = CvBridge()

        # Cropping. 0.5 keeps the bottom half of the image.
        self.crop_fraction = rospy.get_param("~crop_fraction", 0.5)

        # Manual HSV filter. White = low saturation, high value. Hue does not matter.
        self.hsv_lower = tuple(rospy.get_param("~hsv_lower", [0, 0, 140]))
        self.hsv_upper = tuple(rospy.get_param("~hsv_upper", [180, 80, 255]))
        self.erode_size = rospy.get_param("~erode_size", 0)
        self.dilate_size = rospy.get_param("~dilate_size", 5)

        self.pub_cropped = rospy.Publisher("image_cropped", Image, queue_size=10)
        self.pub_white = rospy.Publisher("image_white", Image, queue_size=10)

        # Nodes start at different times, so give publishers a moment to connect
        rospy.sleep(1.0)
        rospy.Subscriber("image", Image, self.callback, queue_size=1)
        rospy.loginfo("ex6 image_processor ready")

    @staticmethod
    def _kernel(size):
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    def callback(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # One shared header for both outputs. The stamp lets message_filters
        # in Node 2 match the cropped image with the mask made from it.
        header = msg.header
        header.stamp = rospy.Time.now()

        # Crop: keep only the bottom part where the lanes are
        height = cv_img.shape[0]
        start_row = int(height * self.crop_fraction)
        cropped = cv_img[start_row:, :]

        # HSV white filter -> binary mask (255 = white, 0 = everything else)
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        if self.erode_size > 0:
            white_mask = cv2.erode(white_mask, self._kernel(self.erode_size))
        if self.dilate_size > 0:
            white_mask = cv2.dilate(white_mask, self._kernel(self.dilate_size))

        cropped_msg = self.bridge.cv2_to_imgmsg(cropped, "bgr8")
        cropped_msg.header = header
        mask_msg = self.bridge.cv2_to_imgmsg(white_mask, "mono8")
        mask_msg.header = header

        self.pub_cropped.publish(cropped_msg)
        self.pub_white.publish(mask_msg)


if __name__ == "__main__":
    rospy.init_node("image_processor")
    ImageProcessor()
    rospy.spin()
