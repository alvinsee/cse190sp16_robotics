# mdp implementation needs to go here

import numpy as np
import math as m
import rospy
import heapq
import sys
import copy
import random
import bisect

from read_config import read_config

SOME_L = 1.337

class Cell():
    def __init__(self, position, reward):
        self.position = position
        self.reward   = reward
        self.q_values = {(0,1): 0.0, (0,-1): 0.0, (1,0): 0.0, (-1,0): 0.0}
        self.n        = {(0,1): 1, (0,-1): 1, (1,0): 1, (-1,0): 1}
        self.policy   = "N"

class Util():
    def __init__(self, utility):
        self.utility = utility
    def __cmp__(self, other):
        return -cmp(self.utility, other.utility)

def weighted_choice(distribution):
    moves, weights = zip(*distribution)
    total = 0.0
    cummulative_weights = []

    for weight in weights:
        total += weight
        cummulative_weights.append(total)

    r = random.uniform(0.0, 1.0)
    return moves[bisect.bisect(cummulative_weights, r)]

def solve():

    random.seed(0)
    map_graph = {}

    config    = read_config()

    row_size  = config['map_size'][0]
    col_size  = config['map_size'][1]

    walls     = config['walls']
    pits      = config['pits']

    start     = config['start']
    goal      = config['goal']

    move_list = config['move_list']

    max_iterations       = config['max_iterations']
    threshold_difference = config["threshold_difference"]

    reward_for_each_step      = config["reward_for_each_step"]
    reward_for_hitting_wall   = config["reward_for_hitting_wall"]
    reward_for_reaching_goal  = config["reward_for_reaching_goal"]
    reward_for_falling_in_pit = config["reward_for_falling_in_pit"]

    discount_factor = config["discount_factor"]
    learning_rate   = config["learning_rate"]

    prob_move_forward  = config["prob_move_forward"]
    prob_move_left     = config["prob_move_left"]
    prob_move_right    = config["prob_move_right"]
    prob_move_backward = config["prob_move_backward"]

    distribution = [("FORWARD", prob_move_forward), ("LEFT", prob_move_left), ("RIGHT", prob_move_right), ("BACKWARDS", prob_move_backward)]

    ###############
    # Set up grid #
    ###############
    for x in range(0, row_size):
        for y in range(0, col_size):
            map_graph[(x, y)] = Cell((x, y), reward_for_each_step)

    for wall in walls:
        wall_cell        = map_graph[(wall[0], wall[1])]
        wall_cell.policy = "WALL"
        wall_cell.reward = reward_for_hitting_wall

    for pit in pits:
        pit_cell        = map_graph[(pit[0], pit[1])]
        pit_cell.policy = "PIT"
        pit_cell.reward = reward_for_falling_in_pit

    goal_cell        = map_graph[tuple(goal)]
    goal_cell.policy = "GOAL"
    goal_cell.reward = reward_for_reaching_goal

    policies = {(0,1):"E", (0,-1):"W", (1,0):"S", (-1,0):"N"}

    ####################
    # Solve Q-Learning #
    ###################
    iteration = 0

    list_policies = []

    curr_cell = map_graph[tuple(start)]

    x = start[0]
    y = start[1]

    while iteration < max_iterations:


        relative_cell = None

        forward    = None
        right      = None
        down       = None
        left       = None

        utility    = 0.0

        action_util = []
        max_q       = []
        neighbours  = []
        policy      = []

        # if pit or goal, reset robot
        if curr_cell.policy == "PIT" or curr_cell.policy == "GOAL":
            x = start[0]
            y = start[1]
            curr_cell = map_graph[tuple(start)]
            #continue

        #print "x: ", x, ", y: ", y
        stationary_cell        = copy.deepcopy(curr_cell)
        stationary_cell.reward = reward_for_hitting_wall

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

        # pick a random action
        action = tuple(move_list[random.randint(0, 3)])

        relative_action = weighted_choice(distribution)

        # Move EAST
        if action == (0,1):
            forward = neighbours[0]
            right   = neighbours[2]
            down    = neighbours[1]
            left    = neighbours[3]
        # Move WEST
        elif action == (0,-1):
            forward = neighbours[1]
            right   = neighbours[3]
            down    = neighbours[0]
            left    = neighbours[2]
        # Move SOUTH
        elif action == (1,0):
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

        if relative_action == "FORWARD":
            for k, q in forward.q_values.iteritems():
                heapq.heappush(action_util, (Util(q + SOME_L/float(forward.n[k])), k))

            relative_cell = forward

            res = heapq.heappop(action_util)
            forward.n[res[1]] += 1

            utility = forward.reward + discount_factor * res[0].utility
            curr_cell.q_values[action] = (1.0 - learning_rate) * curr_cell.q_values[action] + learning_rate * utility

        elif relative_action == "LEFT":
            for k, q in left.q_values.iteritems():
                heapq.heappush(action_util, (Util(q + SOME_L/float(left.n[k])), k))

            relative_cell = left

            res = heapq.heappop(action_util)
            left.n[res[1]] += 1

            utility = left.reward + discount_factor * res[0].utility
            curr_cell.q_values[action] = (1.0 - learning_rate) * curr_cell.q_values[action] + learning_rate * utility

        elif relative_action == "RIGHT":
            for k, q in right.q_values.iteritems():
                heapq.heappush(action_util, (Util(q + SOME_L/float(right.n[k])), k))

            relative_cell = right

            res = heapq.heappop(action_util)
            right.n[res[1]] += 1

            utility = right.reward + discount_factor * res[0].utility
            curr_cell.q_values[action] = (1.0 - learning_rate) * curr_cell.q_values[action] + learning_rate * utility

        else:
            for k, q in down.q_values.iteritems():
                heapq.heappush(action_util, (Util(q + SOME_L/float(down.n[k])), k))

            relative_cell = down

            res = heapq.heappop(action_util)
            down.n[res[1]] += 1

            utility = down.reward + discount_factor * res[0].utility
            curr_cell.q_values[action] = (1.0 - learning_rate) * curr_cell.q_values[action] + learning_rate * utility

        for k, v in curr_cell.q_values.iteritems():
            heapq.heappush(max_q, (Util(v), k))

        curr_cell.policy = policies[(heapq.heappop(max_q)[1])]

        curr_cell = relative_cell
        x = curr_cell.position[0]
        y = curr_cell.position[1]

        #list_policies.append(copy.deepcopy(policy))

        iteration += 1

    for x in range(0, row_size):
        for y in range(0, col_size):
            list_policies.append(map_graph[(x, y)].policy)

    return list_policies
