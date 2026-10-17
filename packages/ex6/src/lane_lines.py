#!/usr/bin/env python3
"""
Exercise 6, Node 2 (lane lines node).

  1. Subscribe to image_cropped and image_white, synced with message_filters
  2. Canny edge detection on the cropped image
  3. bitwise_and the edges with the white mask, publish on image_edges
  4. Probabilistic Hough transform on that combined image
  5. Draw the lines on the cropped image, publish on image_lines_white
"""
import rospy
import cv2
import numpy as np
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class LaneLines:
    def __init__(self):
        self.bridge = CvBridge()

        # Canny thresholds
        self.canny_low = rospy.get_param("~canny_low", 50)
        self.canny_high = rospy.get_param("~canny_high", 150)
        self.blur_size = rospy.get_param("~blur_size", 5)

        # Dilate the edges a little so they overlap the white mask better
        self.edge_dilate = rospy.get_param("~edge_dilate", 3)

        # Hough parameters
        self.hough_threshold = rospy.get_param("~hough_threshold", 20)
        self.min_line_length = rospy.get_param("~min_line_length", 15)
        self.max_line_gap = rospy.get_param("~max_line_gap", 10)

        self.pub_edges = rospy.Publisher("image_edges", Image, queue_size=10)
        self.pub_lines = rospy.Publisher("image_lines_white", Image, queue_size=10)

        # Nodes start at different times, so give publishers a moment to connect
        rospy.sleep(1.0)

        sub_cropped = message_filters.Subscriber("image_cropped", Image)
        sub_white = message_filters.Subscriber("image_white", Image)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [sub_cropped, sub_white], queue_size=10, slop=0.1
        )
        self.sync.registerCallback(self.callback)
        rospy.loginfo("ex6 lane_lines ready")

    def output_lines(self, original_image, lines):
        output = np.copy(original_image)
        if lines is not None:
            for i in range(len(lines)):
                l = lines[i][0]
                cv2.line(output, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 2, cv2.LINE_AA)
                cv2.circle(output, (l[0], l[1]), 2, (0, 255, 0))
                cv2.circle(output, (l[2], l[3]), 2, (0, 0, 255))
        return output

    def callback(self, cropped_msg, white_msg):
        cropped = self.bridge.imgmsg_to_cv2(cropped_msg, "bgr8")
        white_mask = self.bridge.imgmsg_to_cv2(white_msg, "mono8")

        # Canny needs grayscale. A light blur cuts down on noisy edges.
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        if self.blur_size > 1:
            gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        if self.edge_dilate > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.edge_dilate, self.edge_dilate)
            )
            edges = cv2.dilate(edges, kernel)

        # Keep only edges that sit on white lane pixels
        combined = cv2.bitwise_and(edges, white_mask)

        edges_msg = self.bridge.cv2_to_imgmsg(combined, "mono8")
        edges_msg.header = cropped_msg.header
        self.pub_edges.publish(edges_msg)

        # Probabilistic Hough transform
        lines = cv2.HoughLinesP(
            combined,
            1,
            np.pi / 180,
            self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )
        if lines is None:
            rospy.logwarn_throttle(5, "HoughLinesP found no lines")

        lines_img = self.output_lines(cropped, lines)
        lines_msg = self.bridge.cv2_to_imgmsg(lines_img, "bgr8")
        lines_msg.header = cropped_msg.header
        self.pub_lines.publish(lines_msg)


if __name__ == "__main__":
    rospy.init_node("lane_lines")
    LaneLines()
    rospy.spin()
