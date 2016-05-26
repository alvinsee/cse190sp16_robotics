# mdp implementation needs to go here

import numpy as np
import math as m
import rospy
import heapq
import sys
import copy

from cse_190_assi_3.msg import AStarPath

from read_config import read_config

class Cell():
    def __init__(self, reward):
        self.reward = reward
        self.prev   = 0.0
        self.value  = 0.0
        self.policy = ""

class Util():
    def __init__(self, utility):
        self.utility = utility
    def __cmp__(self, other):
        return -cmp(self.utility, other.utility)

def solve():

    map_graph = {}

    config    = read_config()

    row_size  = config['map_size'][0]
    col_size  = config['map_size'][1]

    walls     = config['walls']
    pits      = config['pits']

    goal      = config['goal']

    move_list = config['move_list']

    max_iterations       = config['max_iterations']
    threshold_difference = config["threshold_difference"]

    reward_for_each_step      = config["reward_for_each_step"]
    reward_for_hitting_wall   = config["reward_for_hitting_wall"]
    reward_for_reaching_goal  = config["reward_for_reaching_goal"]
    reward_for_falling_in_pit = config["reward_for_falling_in_pit"]

    discount_factor = config["discount_factor"]

    prob_move_forward  = config["prob_move_forward"]
    prob_move_left     = config["prob_move_left"]
    prob_move_right    = config["prob_move_right"]
    prob_move_backward = config["prob_move_backward"]

    ###############
    # Set up grid #
    ###############
    for x in range(0, row_size):
        for y in range(0, col_size):
            map_graph[tuple([x, y])] = Cell(reward_for_each_step)

    for wall in walls:
        wall_cell        = map_graph[tuple([wall[0], wall[1]])]
        wall_cell.policy = "WALL"
        wall_cell.reward = reward_for_hitting_wall

    for pit in pits:
        pit_cell        = map_graph[tuple([pit[0], pit[1]])]
        pit_cell.policy = "PIT"
        pit_cell.reward = reward_for_falling_in_pit

    goal_cell        = map_graph[tuple(goal)]
    goal_cell.policy = "GOAL"
    goal_cell.reward = reward_for_reaching_goal

    policies = {(0,1):"E", (0,-1):"W", (1,0):"S", (-1,0):"N"}

    #############
    # Solve MDP #
    #############
    iteration = 0
    threshold = sys.maxint

    policy    = []

    while threshold >= threshold_difference and iteration < max_iterations:

        threshold = 0.0

        for x in range(0, row_size):
            for y in range(0, col_size):

                curr_cell              = map_graph[(x, y)]
                stationary_cell        = copy.deepcopy(curr_cell)
                stationary_cell.reward = reward_for_hitting_wall

                action_util = []
                neighbours  = []

                # dont update if wall, pit, or goal
                if curr_cell.policy == "WALL" or curr_cell.policy == "PIT" or curr_cell.policy == "GOAL":
                    continue

                # pre-compute neightbouring cells
                for move in move_list:
                    new_x    = x + move[0]
                    new_y    = y + move[1]

                    # out of bounds
                    if new_x < 0 or new_x >= row_size or new_y < 0 or new_y >= col_size:
                        neighbours.append(stationary_cell)
                        continue

                    neighbour_cell = map_graph[(new_x, new_y)]

                    # hit a wall
                    if neighbour_cell.policy == "WALL":
                        neighbours.append(stationary_cell)
                    else:
                        neighbours.append(neighbour_cell)

                # compute utility for each action
                for action in move_list:

                    utility    = 0.0
                    forward    = None
                    right      = None
                    down       = None
                    left       = None

                    # Move EAST
                    if action == [0, 1]:
                        forward = neighbours[0]
                        right   = neighbours[2]
                        down    = neighbours[1]
                        left    = neighbours[3]
                    # Move WEST
                    elif action == [0,-1]:
                        forward = neighbours[1]
                        right   = neighbours[3]
                        down    = neighbours[0]
                        left    = neighbours[2]
                    # Move SOUTH
                    elif action == [1,0]:
                        forward = neighbours[2]
                        right   = neighbours[1]
                        down    = neighbours[3]
                        left    = neighbours[0]
                    # Move NORTH
                    else:
                        forward = neighbours[3]
                        right   = neighbours[0]
                        down    = neighbours[2]
                        left    = neighbours[1]

                    utility += prob_move_forward  * (forward.reward + discount_factor * forward.prev)
                    utility += prob_move_right    * (right.reward   + discount_factor * right.prev)
                    utility += prob_move_backward * (down.reward    + discount_factor * down.prev)
                    utility += prob_move_left     * (left.reward    + discount_factor * left.prev)

                    heapq.heappush(action_util, (Util(utility), policies[tuple(action)]))

                move_tuple       = heapq.heappop(action_util)
                curr_cell.value  = move_tuple[0].utility
                curr_cell.policy = move_tuple[1]

        # update prev = current value
        for x in range(0, row_size):
            for y in range(0, col_size):
                curr_cell      = map_graph[(x, y)]
                threshold     += abs(curr_cell.value - curr_cell.prev)
                curr_cell.prev = curr_cell.value

        iteration += 1

    for x in range(0, row_size):
        for y in range(0, col_size):
            policy.append(map_graph[(x, y)].policy)

    return policy
