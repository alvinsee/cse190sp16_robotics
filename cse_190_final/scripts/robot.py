#!/usr/bin/env python

import rospy
import numpy as np
import math as m
import random as r

from std_msgs.msg import Bool
from cse_190_final.msg import PolicyList

import qlearning

class Robot():
    def __init__(self):
        rospy.init_node('robot')

        self.bootstrap()
        rospy.sleep(1)

        q_result   = qlearning.solve()

        for x in range(0, len(q_result)):
            if x % 25 == 0:
                self.q_pub.publish(q_result[x])

        # publish final policy
        self.q_pub.publish(q_result[-1])
        rospy.sleep(1)

        self.sim_complete.publish(True)
        rospy.sleep(1)
        rospy.signal_shutdown("Great Success.")

    def bootstrap(self):
        self.q_pub      = rospy.Publisher("/results/policy_list", PolicyList, queue_size = 200)
        self.sim_complete = rospy.Publisher("/map_node/sim_complete", Bool, queue_size = 1)

if __name__ == '__main__':
    rb = Robot()
