#!/usr/bin/env python3
import rospy
import numpy as np
from duckietown_msgs.msg import Vector2D


def se2(theta_deg, x, y):
    t = np.radians(theta_deg)
    return np.array([[np.cos(t), -np.sin(t), x],
                     [np.sin(t),  np.cos(t), y],
                     [0.0,        0.0,       1.0]])


class CoordTransform:
    def __init__(self):
        # Sensor pose in robot frame, robot pose in world frame
        self.T_rs = se2(180, -1, 0)
        self.T_wr = se2(135, 5, 3)
        self.T_ws = self.T_wr @ self.T_rs

        self.pub_robot = rospy.Publisher("robot_coord", Vector2D, queue_size=10)
        self.pub_world = rospy.Publisher("world_coord", Vector2D, queue_size=10)
        rospy.sleep(1.0)  # give publishers time to register
        rospy.Subscriber("sensor_coord", Vector2D, self.callback, queue_size=10)

    def callback(self, msg):
        p_s = np.array([msg.x, msg.y, 1.0])
        p_r = self.T_rs @ p_s
        p_w = self.T_ws @ p_s

        self.pub_robot.publish(Vector2D(x=p_r[0], y=p_r[1]))
        self.pub_world.publish(Vector2D(x=p_w[0], y=p_w[1]))


if __name__ == "__main__":
    rospy.init_node("coord_transform")
    CoordTransform()
    rospy.spin()