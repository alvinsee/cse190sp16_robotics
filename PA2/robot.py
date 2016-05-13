#!/usr/bin/env python

import rospy
import numpy as np
import math as m
import random as r
import helper_functions as hf

import copy
from read_config import read_config

from std_msgs.msg import Bool
from nav_msgs.msg import OccupancyGrid, MapMetaData
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Twist
from sklearn.neighbors import KDTree
from map_utils import Map

class Particle():
    def __init__(self,x,y,t,w):
        self.x = x
        self.y = y
        self.theta = t
        self.weight = w

class Robot():
    def __init__(self):
        rospy.init_node('robot')
        self.config = read_config()
        r.seed(self.config["seed"])

        self.rate = rospy.Rate(1)
        self.num_moves = 0
        self.moved = False

        self.bootstrap()
        rospy.spin()

    def bootstrap(self):
        rospy.Subscriber("/map", OccupancyGrid, self.handle_mapserver)
        rospy.Subscriber("base_scan_with_error", LaserScan, self.handle_scan)
        self.particle_pub = rospy.Publisher("/particlecloud", PoseArray, queue_size = 10)
        self.likelihood_pub = rospy.Publisher("/likelihood_field", OccupancyGrid, queue_size = 10, latch = True)
        self.result_update = rospy.Publisher("/result_update", Bool, queue_size = 1)
        self.sim_complete = rospy.Publisher("/sim_complete", Bool, queue_size = 1)


    def update_likelihood(self):
        obstacle_loc = []
        for x in xrange(self.width):
            for y in xrange(self.height):
                if self.map.get_cell(x,y) == 1.0:
                    obstacle_loc.append([x,y])

        self.kdt = KDTree(obstacle_loc)

        for x in xrange(self.width):
            for y in xrange(self.height):
                dist, index = self.kdt.query([x,y], k = 1)
                val = self.normpdf(0, self.config["laser_sigma_hit"], dist[0])
                self.likelihood_map.set_cell(x, y, val)

        self.likelihood_pub.publish(self.likelihood_map.to_message())


    def initialize_particles(self):
        self.size = self.config["num_particles"]
        self.particles = []
        poses = []

        for i in xrange(self.size):
            x = r.randint(0,self.width)
            y = r.randint(0,self.height)
            t = r.random()*2*m.pi

            w = 1./self.size
            poses.append(hf.get_pose(x,y,t))
            self.particles.append(Particle(x,y,t,w))

        self.publish_pose(poses)
        self.rate.sleep()

    def move(self):
        num_total_moves = len(self.config["move_list"])
        sdt = self.config["first_move_sigma_angle"]
        sdx = self.config["first_move_sigma_x"]
        sdy = self.config["first_move_sigma_y"]

        while(self.num_moves < num_total_moves):
            move_list = self.config["move_list"][self.num_moves]
            self.result_update.publish(True)
            a = move_list[0]
            d = move_list[1]
            n = move_list[2]
            #self.rate.sleep()
            hf.move_function(a, 0)
            a *= m.pi/180.0
            if self.num_moves == 0:
                a += r.gauss(0,sdt)

            for i in xrange(self.size):
                self.particles[i].theta += a

            # Wait til laser scan update finishes
            self.moved = True
            while(self.moved):
                continue

            self.rate.sleep()
            for step in xrange(n):
                hf.move_function(0,d)
                for i in xrange(self.size):
                    if self.num_moves == 0:
                        dx = d*m.cos(self.particles[i].theta) + r.gauss(0,sdx)
                        dy = d*m.sin(self.particles[i].theta) + r.gauss(0,sdy)
                    else:
                        dx = d*m.cos(self.particles[i].theta)
                        dy = d*m.sin(self.particles[i].theta)

                    self.particles[i].x += dx
                    self.particles[i].y += dy

                self.moved = True
                while(self.moved):
                    continue
                self.rate.sleep()

            self.num_moves += 1

        self.sim_complete.publish(True)
        rospy.sleep(1)
        rospy.signal_shutdown("Done.")


    def publish_pose(self,poses):
        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = 'map'
        pose_array.poses = poses
        self.particle_pub.publish(pose_array)

    def handle_scan(self, resp):
        if self.moved == True:
            total_weight = 0.0
            for i in xrange(self.size):
                coord = self.map.get_cell(self.particles[i].x, self.particles[i].y)
                if(coord == 1.0 or coord != coord): #float('nan')):
                    self.particles[i].weight = 0.0
                else:
                    Ptot = 0.0
                    inc = 0
                    for d in resp.ranges:
                        if(d < resp.range_min or d > resp.range_max):
                            continue
                        angle = resp.angle_min + inc * resp.angle_increment + self.particles[i].theta
                        inc += 1
                        x = self.particles[i].x + d*m.cos(angle)
                        y = self.particles[i].y + d*m.sin(angle)


                        Lp = self.likelihood_map.get_cell(x,y)
                        if(Lp != Lp):#or Lp == 1.0):
                            pz = 0
                        else:
                            pz = self.config["laser_z_hit"]*Lp + self.config["laser_z_rand"]
                        Ptot += pz

                    if Ptot == 0:
                        self.particles[i].weight = 0.0
                    else:
                        self.particles[i].weight *= self.sigmoid(Ptot) + 0.01
                        total_weight += self.particles[i].weight

            if(total_weight > 0):
                wheel = []
                particles = []
                counter = 0
                for i in xrange(self.size):
                    self.particles[i].weight /= total_weight
                    if(self.particles[i].weight > 0.0):
                        counter += 1
                    wheel.append(self.particles[i].weight)
                    particles.append(copy.deepcopy(self.particles[i]))

                self.resample(particles, wheel)

            self.moved = False

    def resample(self, particles, wheel):
        poses = []
        sdx = self.config['resample_sigma_x']
        sdy = self.config['resample_sigma_y']
        sdt = self.config['resample_sigma_angle']
        new_p = np.random.choice(particles, self.size, p = wheel)

        for i in xrange(self.size):
            self.particles[i].x = new_p[i].x + r.gauss(0,sdx)
            self.particles[i].y = new_p[i].y + r.gauss(0,sdy)
            self.particles[i].theta = new_p[i].theta + r.gauss(0,sdt)
            self.particles[i].weight = new_p[i].weight

            poses.append(hf.get_pose(self.particles[i].x,self.particles[i].y,self.particles[i].theta))

        self.publish_pose(poses)

    def handle_mapserver(self, resp):
        self.map = Map(resp)
        self.likelihood_map = Map(resp)

        self.width = self.map.width
        self.height = self.map.height

        self.initialize_particles()
        self.update_likelihood()
        self.move()

    def normpdf(self, mean, sd, x):
        var = float(sd)**2
        return m.exp(-(float(x)-float(mean))**2/(2*var))

    def sigmoid(self, p):
        return 1/(1+m.exp(-p))
if __name__ == '__main__':
    rb = Robot()
