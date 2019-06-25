####################
# Contains the class for an puntiform holonomic agent
# State: x,y,theta, dist sx, dist dx
####################

import numpy as np

def mktr(x, y):
    return np.array([[1,0,x],
        [0,1,y],
        [0,0,1]])

def mkrot(theta):   # non serve
    return np.array([[np.cos(theta),-np.sin(theta),0],
        [np.sin(theta), np.cos(theta),0],
        [0,0,1]])

def atr(v,dt):
    return mktr(v*dt,0)  # note we translate along x

class Agent:
    def __init__(self, initial_pose, number):
        self.pose = initial_pose # will always store the current pose
        self.number = number
        self.state = np.array([0,0]) #distanza destra e sinistra 
        
    def step(self, v, dt):
        self.pose = np.matmul(self.pose, atr(v, dt)) # this is where the magic happens
        return self.pose

    def getxy(self):
        return self.pose[0:2,2] # extracts [x,y] from current pose

    def gettheta(self): # sin(theta)      cos(theta)
        return np.arctan2(self.pose[1,0], self.pose[0,0])

    def observe(self, agents_list, L, dt):
        right = L
        left = L

        for a in agents_list:
            dist = self.pose[0:2,2][0] - a.getxy()[0]
            if dist > 0:
                if dist < left:
                    left = dist
            if dist < 0:
                if abs(dist)<right:
                    right= abs(dist)
        if right == L:
            right = L - self.pose[0:2,2][0]
        if left == L:
            left = self.pose[0:2,2][0]

        d_left = (left - self.state[0])/dt
        d_right = (right - self.state[1])/dt

        return np.array([left,right]), np.array([d_left,d_right])