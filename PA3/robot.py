#!/usr/bin/env python

import rospy
import numpy as np
import math as m
import random as r

from std_msgs.msg import Bool
from cse_190_assi_3.msg import AStarPath
from cse_190_assi_3.msg import PolicyList

import astar
import mdp

class Robot():
    def __init__(self):
        rospy.init_node('robot')

        self.bootstrap()
        rospy.sleep(1)

        astar_result = astar.solve()

        for result in astar_result:
            self.astar_pub.publish(result)

        mdp_result   = mdp.solve()
        self.mdp_pub.publish(mdp_result)

        self.sim_complete.publish(True)
        rospy.sleep(1)
        rospy.signal_shutdown("Great Success.")

    def bootstrap(self):
        self.astar_pub    = rospy.Publisher("/results/path_list", AStarPath, queue_size = 50)
        self.mdp_pub      = rospy.Publisher("/results/policy_list", PolicyList, queue_size = 10)
        self.sim_complete = rospy.Publisher("/map_node/sim_complete", Bool, queue_size = 1)

if __name__ == '__main__':
    rb = Robot()
