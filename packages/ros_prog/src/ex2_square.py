#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist

class SquareMove:
    def __init__(self):
        self.pub = rospy.Publisher('turtle1/cmd_vel', Twist, queue_size=10)
        self.rate = rospy.Rate(10)  # 10 Hz

    def move_straight(self, speed, duration):
        cmd = Twist()
        cmd.linear.x = speed
        start = rospy.Time.now()
        while (rospy.Time.now() - start).to_sec() < duration and not rospy.is_shutdown():
            self.pub.publish(cmd)
            self.rate.sleep()
        self.stop()

    def turn(self, angular_speed, duration):
        cmd = Twist()
        cmd.angular.z = angular_speed
        start = rospy.Time.now()
        while (rospy.Time.now() - start).to_sec() < duration and not rospy.is_shutdown():
            self.pub.publish(cmd)
            self.rate.sleep()
        self.stop()

    def stop(self):
        self.pub.publish(Twist())  # all zeros

    def draw_square(self):
        side_speed = 1.0
        side_duration = 2.0
        turn_speed = 1.0
        turn_duration = 1.57  # ~90 degrees at 1 rad/s

        for _ in range(4):
            self.move_straight(side_speed, side_duration)
            rospy.sleep(0.3)
            self.turn(turn_speed, turn_duration)
            rospy.sleep(0.3)

if __name__ == '__main__':
    try:
        rospy.init_node('square_move', anonymous=True)
        rospy.sleep(1.0)
        sq = SquareMove()
        sq.draw_square()
    except rospy.ROSInterruptException:
        pass