#import rospy
import numpy as np
#np.set_printoptions(precision = 4, suppress = True)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random as rand

def mktr(x, y):
	return np.array([[1,0,x],
		[0,1,y],
		[0,0,1]])

def mkrot(theta):	# non serve
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
	#	self.rate = rospy.Rate(10)

	def step(self, v, dt):
		self.pose = np.matmul(self.pose, atr(v, dt)) # this is where the magic happens
		return self.pose

	def getxy(self):
		return self.pose[0:2,2] # extracts [x,y] from current pose

	def gettheta(self): # sin(theta)      cos(theta)
		return np.arctan2(self.pose[1,0], self.pose[0,0])

	def observe(self, agents_list, L):
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

		return np.array([left,right])

class PID:

    def __init__(self, Kp, Ki, Kd):
    	self.Kp = Kp
    	self.Ki = Ki
    	self.Kd = Kd
    	self.last_e = None
    	self.sum_e = 0

    def step(self, e, dt):
    	""" dt should be the time interval from the last method call """
    	if(self.last_e is not None):
    		derivative = (e - self.last_e) / dt
    	else:
    		derivative = 0
    	self.last_e = e
    	self.sum_e += e * dt
    	return self.Kp * e + self.Kd * derivative + self.Ki * self.sum_e

def run_simulation(timesteps, n_agents, L):
    	states = np.zeros((timesteps,n_agents,5))
    	target_vels = np.zeros((timesteps, n_agents))
    	agents_list = [None] * n_agents

    	### Initialize agents at random
    	for i in range(n_agents):
    		initial_pose = np.matmul(np.eye(3), mktr(rand.random()*L, 0))
    		agents_list[i]=Agent(initial_pose,i)
    	for i in range(n_agents):
    		agents_list[i].state = agents_list[i].observe(agents_list,L)
    	# Gli ordino in base alla posizione
    	agents_list.sort(key=lambda x: x.getxy()[0], reverse=False)

    	#save initial state
    	states[0]= save_state(agents_list)

    	# Controller onniscente
    	goal_list = [None]*n_agents
    	dist = L/(n_agents+1)
    	for i in range(n_agents):
    		goal_list[i]=dist*(i+1)

    	controller = PID(-0.8,0,0)

    	#Parameters
    	mas_vel = 1 # m/s
    	dt= 0.1
    	for t in range (1, timesteps):
    		for i in range( 0, len(agents_list)):
    			error = agents_list[i].getxy()[0]-goal_list[i]
    			v = controller.step(error, dt)
    			v = np.clip(v, -mas_vel, +mas_vel)
    			agents_list[i].step(v,dt)
    			target_vels[t,i] = v
    		for i in range( 0, len(agents_list)):
    			agents_list[i].state = agents_list[i].observe(agents_list,L)
    		states[t] = save_state(agents_list)

    	return states, target_vels



def save_state(agents_list):
	n_agents=len(agents_list)
	state = np.zeros((n_agents, 5))
	for i in range(n_agents):

		state[i,0]= agents_list[i].getxy()[0]
		state[i,1]= agents_list[i].getxy()[1]
		state[i,2]= agents_list[i].gettheta()
		state[i,3]= agents_list[i].state[0]
		state[i,4]= agents_list[i].state[1]
	return state

def plot_simulation(data, L):

	numframes = len(data)-1
	numpoints = len(data[0])
	x = data[0,:,0]
	y = data[0,:,1]

	x_data = data[1:,:,0]
    
	y_data = data[1:,:,1]

	fig = plt.figure()
	

	#data = np.random.rand(numframes, 2,5)
	l, = plt.plot([], [], 'r-')
	plt.xlim(0, L)
	plt.ylim(-1, 1)
	plt.xlabel('x')
	plt.title('Task 1')


	line = np.linspace(0, L)
	plt.plot(line, 0*line);

	scat = plt.scatter(x, y, s=100)
#	points = plt.plot([1,2,3,4,5], [0,0,0,0,0], 'ro')


	for i in xrange(numframes):
		ani = animation.FuncAnimation(fig, update_plot ,frames = numframes ,fargs=(x_data,y_data, scat))


	plt.show()	

def update_plot(i, x_data,y_data, scat):
	data = np.array([x_data[i],y_data[i]]).T
	scat.set_offsets(data)
	return scat,
		





##########################################################################


if __name__ == '__main__':
	L = 10. # Distanza tra i due muri
	N = 5 # Number of agents
	timesteps = 100 #timesteps

	states, target_vels = run_simulation(timesteps,N,L)

	plot_simulation(states,L)
	print('ok')