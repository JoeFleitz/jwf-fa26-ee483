#!/usr/bin/env python3
import rospy
import math
from turtlesim.msg import Pose
from std_msgs.msg import Float32

METERS_TO_FEET = 3.28084

class DistanceCalculator:
    def __init__(self):
        self.pub = rospy.Publisher('total_distance', Float32, queue_size=10)
        self.last_x = None
        self.last_y = None
        self.total_distance = 0.0
        self.current_unit = None

        rospy.sleep(1)
        rospy.Subscriber('/turtle1/pose', Pose, self.callback)

    def get_unit(self):
        if rospy.has_param("distance_unit"):
            return str(rospy.get_param("distance_unit")).lower()
        return "meters"   # fallback if the parameter doesn't exist

    def callback(self, msg):
        if self.last_x is not None:
            dx = msg.x - self.last_x
            dy = msg.y - self.last_y
            self.total_distance += math.sqrt(dx**2 + dy**2)

        self.last_x = msg.x
        self.last_y = msg.y

        unit = self.get_unit()
        if unit != self.current_unit:
            rospy.loginfo("Set distance unit to: %s", unit)
            self.current_unit = unit

        if unit == "feet":
            distance = self.total_distance * METERS_TO_FEET
        else:
            distance = self.total_distance

        self.pub.publish(self.total_distance)
        rospy.loginfo("Total distance: %.2f %s", distance, unit)

if __name__ == '__main__':
    try:
        rospy.init_node('distance_calculator', anonymous=True)
        DistanceCalculator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass