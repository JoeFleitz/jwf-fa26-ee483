#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32
from pid import PID


class PIDControllerNode:
    def __init__(self):
        # Gains (tune these)
        kp = 1.5
        ki = 0.0
        kd = 0.05

        self.pid = PID(kp, ki, kd, integral_limit=1.0, output_limit=None)

        self.pub = rospy.Publisher("control_input", Float32, queue_size=10)
        rospy.Subscriber("error", Float32, self.error_callback, queue_size=1)

        rospy.loginfo("PID gains kp=%.3f ki=%.3f kd=%.3f", kp, ki, kd)

    def error_callback(self, msg):
        now = rospy.get_time()
        omega = self.pid.update(msg.data, now)
        self.pub.publish(Float32(omega))


if __name__ == "__main__":
    rospy.init_node("pid_controller", anonymous=False)
    node = PIDControllerNode()

    # Give the other nodes a moment to come up and connect before the sim starts
    rospy.sleep(2.0)
    rospy.set_param("controller_ready", "true")

    rospy.spin()
