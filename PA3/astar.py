# astar implementation needs to go here

import numpy as np
import math as m
import rospy
import heapq

from read_config import read_config

class Node():
    def __init__(self):
        self.prev          = None
        self.total_cost    = 0
        self.explored      = False
        self.pos           = None

def solve():

    map_graph = {}

    config    = read_config()

    row_size  = config['map_size'][0]
    col_size  = config['map_size'][1]

    walls     = config['walls']
    pits      = config['pits']

    start     = config['start']
    goal      = config['goal']

    move_list = config['move_list']

    ###############
    # Set up grid #
    ###############
    for x in range(0, row_size):
        for y in range(0, col_size):
            map_graph[tuple([x, y])] = Node()

    for wall in walls:
        map_graph[tuple([wall[0], wall[1]])].total_cost = -1000

    for pit in pits:
        map_graph[tuple([pit[0], pit[1]])].total_cost   = -1000

    for x in range(0, row_size):
        for y in range(0, col_size):
            if map_graph[tuple([x, y])].total_cost == -1000:
                continue
            curr_node            = map_graph[tuple([x, y])]
            forward_cost         = abs(goal[0]-x) + abs(goal[1]-y)
            backward_cost        = abs(start[0]-x) + abs(start[1]-y)
            curr_node.total_cost = forward_cost + backward_cost
            curr_node.pos        = tuple([x, y])

    ############
    # Solve A* #
    ############
    move_count = 0
    move_queue = []
    path = []

    start_node = map_graph[tuple(start)]
    goal_pos   = tuple(goal)
    goal_node  = map_graph[goal_pos]

    heapq.heappush(move_queue, (start_node.total_cost, tuple(start)))

    while len(move_queue) > 0:

        move_tuple = heapq.heappop(move_queue)
        total_cost = move_tuple[0]
        curr_pos   = move_tuple[1]
        curr_node  = map_graph[curr_pos]

        if(curr_pos == goal_pos):
            prev_node = goal_node

            while prev_node != None:
                path.insert(0, list(prev_node.pos))
                prev_node = prev_node.prev

            return path

        x = curr_pos[0]
        y = curr_pos[1]

        for move in move_list:
            new_x    = x + move[0]
            new_y    = y + move[1]
            next_pos = tuple([new_x, new_y])

            if new_x < 0 or new_x >= row_size or new_y < 0 or new_y >= col_size:
                continue

            adj_node = map_graph[next_pos]

            if adj_node.explored or adj_node.total_cost == -1000:
                continue

            curr_node.explored = True
            adj_node.prev      = curr_node

            heapq.heappush(move_queue, (adj_node.total_cost, next_pos))

    return []
