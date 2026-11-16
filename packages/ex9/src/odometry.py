#!/usr/bin/env python3
import math
import rospy
from odom_aux.msg import DistWheel, Pose2D


class Odometry:
    def __init__(self):
        # Robot starts at (0, 0, 0)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.L = 0.05  # half the wheel baseline (2L = 0.1 m)

        self.pub = rospy.Publisher("pose", Pose2D, queue_size=10)
        rospy.sleep(1.0)
        rospy.Subscriber("dist_wheel", DistWheel, self.callback, queue_size=10)

    def callback(self, msg):
        ds_l = msg.dist_wheel_left
        ds_r = msg.dist_wheel_right

        # Distance traveled by the robot center and change in heading
        ds = (ds_l + ds_r) / 2.0
        dtheta = (ds_r - ds_l) / (2.0 * self.L)

        # Use the midpoint heading for the position update
        sigma = self.theta + dtheta / 2.0
        self.x += ds * math.cos(sigma)
        self.y += ds * math.sin(sigma)
        self.theta += dtheta

        # Keep theta in [-pi, pi]
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        pose = Pose2D()
        pose.x = self.x
        pose.y = self.y
        pose.theta = self.theta
        self.pub.publish(pose)


if __name__ == "__main__":
    rospy.init_node("odometry", anonymous=True)
    Odometry()
    rospy.spin()
