#import rospy
import numpy as np
#np.set_printoptions(precision = 4, suppress = True)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random as rand
import tensorflow as tf
from simple_net import Net

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

def run_simulation(timesteps, n_agents, L, init= None):
		states = np.zeros((timesteps,n_agents,5))
		target_vels = np.zeros((timesteps, n_agents))
		agents_list = [None] * n_agents

		### Initialize agents at random if not given initial state
		if init is None:
			for i in range(n_agents):
				initial_pose = np.matmul(np.eye(3), mktr(rand.random()*L, 0))
				agents_list[i]=Agent(initial_pose,i)
		else:
			for i in range(n_agents):
				agents_list[i]=Agent(init[i],i)

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


def run_simulation_learned(timesteps, n_agents, L, Net, init=None):
		states = np.zeros((timesteps,n_agents,5))
		target_vels = np.zeros((timesteps, n_agents))
		agents_list = [None] * n_agents

		### Initialize agents at random if not given initial state
		if init is None:
			for i in range(n_agents):
				initial_pose = np.matmul(np.eye(3), mktr(rand.random()*L, 0))
				agents_list[i]=Agent(initial_pose,i)
		else:
			for i in range(n_agents):
				agents_list[i]=Agent(init[i],i)

		# Gli ordino in base alla posizione
		agents_list.sort(key=lambda x: x.getxy()[0], reverse=False)

		#save initial state
		states[0]= save_state(agents_list)

		#Parameters
		mas_vel = 1 # m/s
		dt= 0.1
		for t in range (1, timesteps):
			inputs = states[t-1,:,3:]
			inputs = inputs.reshape(1, -1)
			predicted_velocities = Net.predict(inputs)
			
			for i in range( 0, len(agents_list)):
#				error = agents_list[i].getxy()[0]-goal_list[i]
				v = predicted_velocities[0,i]
				v = np.clip(v, -mas_vel, +mas_vel)
				agents_list[i].step(v,dt)
			for i in range( 0, len(agents_list)):
				agents_list[i].state = agents_list[i].observe(agents_list,L)
			states[t] = save_state(agents_list)

		return states





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


	for i in range(numframes):
		ani = animation.FuncAnimation(fig, update_plot ,frames = numframes ,fargs=(x_data,y_data, scat))


	plt.show()

def plot_simulation2(data1,data2, L):

	numframes = len(data1)-1
	numpoints = len(data1[0])
	x1 = data1[0,:,0]
	y1 = data1[0,:,1]

	x2 = data2[0,:,0]
	y2 = data2[0,:,1]



	x_data1 = data1[1:,:,0]	
	y_data1 = data1[1:,:,1]

	x_data2 = data2[1:,:,0]	
	y_data2 = data2[1:,:,1]


	fig = plt.figure()
	

	#data = np.random.rand(numframes, 2,5)
	l, = plt.plot([], [], 'r-')
	plt.xlim(0, L)
	plt.ylim(-1, 1)
	plt.xlabel('x')
	plt.title('Task 1')


	line = np.linspace(0, L)
	plt.plot(line, 0*line);

	scat1 = plt.scatter(x1, y1, s=100)
	scat2 = plt.scatter(x2, y2, s=100, c='red')
#	points = plt.plot([1,2,3,4,5], [0,0,0,0,0], 'ro')


	for i in range(numframes):
		ani = animation.FuncAnimation(fig, update_plot ,frames = numframes ,fargs=(x_data1,y_data1,x_data2,y_data2, scat1, scat2))

	ani.save('plots/animation.gif', writer='imagemagick', fps=10)
	plt.show()	

def update_plot(i, x_data1,y_data1, x_data2,y_data2, scat1, scat2):
	data1 = np.array([x_data1[i],y_data1[i]]).T
	scat1.set_offsets(data1)
	data2 = np.array([x_data2[i],y_data2[i]]).T
	scat2.set_offsets(data2)
	
	return scat1, scat2
		





##########################################################################


if __name__ == '__main__':
	L = 10. # Distanza tra i due muri
	N = 5 # Number of agents
	timesteps = 120 #timesteps
	n_simulation = 100 # number of simulations

	net_inputs = None
	net_outputs = None


	for i in range(n_simulation):
		states, target_vels = run_simulation(timesteps,N,L)
		
		# prepara data for the training of the net
		t_data=states[:,:,3:]
		t_data = t_data.reshape(t_data.shape[0], -1)

		if net_inputs is None:
			net_inputs = t_data
			net_outputs=target_vels
		else:
			net_inputs = np.append(net_inputs,t_data,axis=0)
			net_outputs = np.append(net_outputs,target_vels,axis=0)


	#train the net

	Net = Net(net_inputs.shape[1],net_outputs.shape[1],1000,net_inputs.shape[0],1)
	Net.set_batch_data(net_inputs,net_outputs)
	Net.train()


	# Make a confront of the two simulations
	init = []
	for i in range(N):
		init.append(np.matmul(np.eye(3), mktr(rand.random()*L, 0)))

	states, _ =run_simulation(timesteps,N,L,init)
	states_L= run_simulation_learned(timesteps,N,L,Net,init)




	#plot_simulation(states,L)
	#plot_simulation(states_L,L)
	plot_simulation2(states,states_L,L)
	
	print('ok')