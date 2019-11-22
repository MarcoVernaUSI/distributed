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
    def __init__(self, initial_pose, number, min_range = 0.0, max_range = 9999999.0):
        self.pose = initial_pose # will always store the current pose
        self.number = number
        self.state = np.zeros(2) #distanza destra e sinistra (if 2) # velocità dei vicini se 4
        self.vels = np.zeros(2)
        self.width = 0.06 #m
        self.velocity = 0.0
        self.min_range = min_range
        self.max_range = max_range
        self.color = 0
        
    def step(self, v, dt):
        self.pose = np.matmul(self.pose, atr(v, dt))
        self.velocity = v
        return self.pose

    def getxy(self):
        return self.pose[0:2,2] # extracts [x,y] from current pose

    def setx(self, x):
        self.pose[0,2]=x
    def sety(self,y):
        self.pose[1,2]=y 

    def distance(self, agent):
        if isinstance(agent, Agent):
            return self.pose[0:2,2][0] - agent.getxy()[0]
        else:
            return self.pose[0:2,2][0] - agent

    def check_collisions(self, agents_list, L):
        if (self.distance(0) < self.width*2) and (self.distance(L)> self.width*2):
            return False
        for a in agents_list:
            if abs(self.distance(a))<self.width*2:
                return False
        return True

    def gettheta(self): # sin(theta)      cos(theta)
        return np.arctan2(self.pose[1,0], self.pose[0,0])

    def observe(self, agents_list, L, idx):
        right = L
        left = L

        vel_right = 0.0
        vel_left = 0.0

        for i,a in enumerate(agents_list):
            dist = self.distance(a)
            if dist > 0 and i!=idx:
                if dist < left:
                    left = dist
                    vel_left = a.velocity
            if dist < 0 and i!=idx:
                if abs(dist)<right:
                    right= abs(dist)
                    vel_right=a.velocity
        if right == L:
            right = -self.distance(L)
            right= right - self.width
        else:
            right=right-(2*self.width)

        # adjust with width
        if left == L:
            left = self.distance(0)
            left = left - self.width
        else:
            left=left-(2*self.width)
        
        return np.array([np.clip(left, self.min_range,self.max_range),np.clip(right, self.min_range,self.max_range)]), np.array([vel_left,vel_right])