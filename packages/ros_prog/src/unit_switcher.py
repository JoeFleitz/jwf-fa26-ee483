#!/usr/bin/env python3
import rospy

if __name__ == '__main__':
    try:
        rospy.init_node('unit_switcher', anonymous=True)
        rospy.sleep(1)   # give the other nodes time to start

        while not rospy.is_shutdown():
            rospy.sleep(5)
            rospy.set_param("distance_unit", "Meters")
            rospy.sleep(5)
            rospy.set_param("distance_unit", "Feet")
    except rospy.ROSInterruptException:
        pass