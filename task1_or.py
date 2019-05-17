#import rospy
import numpy as np
#np.set_printoptions(precision = 4, suppress = True)
from matplotlib import pyplot, transforms
from matplotlib.lines import Line2D
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
		self.state_d = np.array([0,0]) #e loro derivata
	#	self.rate = rospy.Rate(10)

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
		states = np.zeros((timesteps,n_agents,7))    ############# 5 -> 7
		target_vels = np.zeros((timesteps, n_agents))
		agents_list = [None] * n_agents
		dt= 0.1

		### Initialize agents at random if not given initial state
		if init is None:
			for i in range(n_agents):
				initial_pose = np.matmul(np.eye(3), mktr(rand.random()*L, 0))
				agents_list[i]=Agent(initial_pose,i)
		else:
			for i in range(n_agents):
				agents_list[i]=Agent(init[i],i)

		for i in range(n_agents):
			agents_list[i].state, agents_list[i].state_d  = agents_list[i].observe(agents_list,L,dt)
		# Gli ordino in base alla posizione
		agents_list.sort(key=lambda x: x.getxy()[0], reverse=False)

		#save initial state
		states[0]= save_state(agents_list)

		# Controller onniscente
		goal_list = [None]*n_agents
		dist = L/(n_agents+1)
		for i in range(n_agents):
			goal_list[i]=dist*(i+1)

		controller = PID(-5,0,0)

		#Parameters
		mas_vel = 2 # m/s
		
		for t in range (1, timesteps):
			for i in range( 0, len(agents_list)):
				error = agents_list[i].getxy()[0]-goal_list[i]
				v = controller.step(error, dt)
				v = np.clip(v, -mas_vel, +mas_vel)
				agents_list[i].step(v,dt)
				target_vels[t,i] = v
			for i in range( 0, len(agents_list)):
				agents_list[i].state, agents_list[i].state_d = agents_list[i].observe(agents_list,L,dt)
			states[t] = save_state(agents_list)

		return states, target_vels


def run_simulation_learned(timesteps, n_agents, L, Net, init=None):
		states = np.zeros((timesteps,n_agents,7))
		target_vels = np.zeros((timesteps, n_agents))
		agents_list = [None] * n_agents
		dt= 0.1

		### Initialize agents at random if not given initial state
		if init is None:
			for i in range(n_agents):
				initial_pose = np.matmul(np.eye(3), mktr(rand.random()*L, 0))
				agents_list[i]=Agent(initial_pose,i)
		else:
			for i in range(n_agents):
				agents_list[i]=Agent(init[i],i)


		for i in range(n_agents):
			agents_list[i].state, agents_list[i].state_d  = agents_list[i].observe(agents_list,L,dt)
		
		# Gli ordino in base alla posizione
		agents_list.sort(key=lambda x: x.getxy()[0], reverse=False)

		#save initial state
		states[0]= save_state(agents_list)

		#Parameters
		mas_vel = 1 # m/s
		
		for t in range (1, timesteps):

			inputs = states[t-1,:,3:5]
			inputs = inputs.reshape(1, -1)
			#predicted_velocities = Net.predict(inputs)

			
			for i in range( 0, len(agents_list)):
#				error = agents_list[i].getxy()[0]-goal_list[i]
				
				if isinstance(Net, (list,)):
					predicted_velocities = Net[i].predict(np.array([inputs[0,2*i:2*i+2]]))
					v = predicted_velocities[0,0]
				else:
					predicted_velocities = Net.predict(inputs)
					v = predicted_velocities[0,i]
				
				v = np.clip(v, -mas_vel, +mas_vel)
				agents_list[i].step(v,dt)
			for i in range( 0, len(agents_list)):
				agents_list[i].state, agents_list[i].state_d = agents_list[i].observe(agents_list,L, dt)
			states[t] = save_state(agents_list)

		return states


def save_state(agents_list):
	n_agents=len(agents_list)
	state = np.zeros((n_agents, 7))
	for i in range(n_agents):

		state[i,0]= agents_list[i].getxy()[0]
		state[i,1]= agents_list[i].getxy()[1]
		state[i,2]= agents_list[i].gettheta()
		state[i,3]= agents_list[i].state[0]
		state[i,4]= agents_list[i].state[1]
		state[i,5]= agents_list[i].state_d[0]
		state[i,6]= agents_list[i].state_d[1]
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



def timeGraph(data1,data2, L):



	x_data1 = data1[0:,:,0]	
	y_data1 = data1[0:,:,1]

	x_data2 = data2[0:,:,0]	
	y_data2 = data2[0:,:,1]


	fig = plt.figure()

	base = pyplot.gca().transData
	rot = transforms.Affine2D().rotate_deg(90)

	plt.plot(x_data1,color='blue',label = 'optimal',transform= rot + base)
	plt.plot(x_data2,color='red',label = 'learned',transform= rot + base)
	
	custom_lines = [Line2D([0], [0], color='blue', lw=4),Line2D([0], [0], color='red', lw=4)]
	plt.legend(custom_lines, ['Optimal', 'Learned'],loc=4)

	plt.xlabel('x position')
	plt.ylabel('timesteps')
	plt.title('Task 1')


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
	n_simulation = 1000 # number of simulations

	net_inputs = None
	net_outputs = None


	for i in range(n_simulation):
		states, target_vels = run_simulation(timesteps,N,L)
		
		# prepara data for the training of the net

		t_data=states[:,:,3:5]

		t_data = t_data.reshape(t_data.shape[0], -1)

		if net_inputs is None:
			net_inputs = t_data
			net_outputs=target_vels
		else:
			net_inputs = np.append(net_inputs,t_data,axis=0)
			net_outputs = np.append(net_outputs,target_vels,axis=0)


	#train the global net
#	Net = Net(net_inputs.shape[1],net_outputs.shape[1],1000,net_inputs.shape[0],1,0.12)
#	Net.set_batch_data(net_inputs,net_outputs)
#	Net.train()

	# train one net for each agent


#	NetA = Net(net_inputs.shape[1]/N,net_outputs.shape[1]/N,500,net_inputs.shape[0],1,0.12)
	#NetA.set_batch_data(net_inputs[:,:,2*i:2*i+2],net_outputs[:,:,i].reshape(net_outputs.shape[0],net_outputs.shape[1], 1))
	#NetA.set_batch_data(net_inputs.reshape(:,:,2) , net_outputs[:,:,i].reshape(net_outputs.shape[0],net_outputs.shape[1], 1))
		

	# train one net for each agent
	Nets = []

	for i in range(N):
		NetA = Net(net_inputs.shape[1]/N,net_outputs.shape[1]/N,500,100,1,0.001)
		#NetA.set_batch_data(net_inputs[:,2*i:2*i+2],np.array([net_outputs[:,i]]).reshape(-1,1))
		NetA.set_batch_data(net_inputs.reshape(-1,2),np.array([net_outputs]).reshape(-1,1))
		
		NetA.train(str(i))
		print('train net ',i,' done')
		Nets.append(NetA)



	#NetA.train()
	#print('train net ',i,' done')
	
	# Make a confront of the two simulations
	init = []
	for i in range(N):
		init.append(np.matmul(np.eye(3), mktr(rand.random()*L, 0)))

	states, _ =run_simulation(timesteps,N,L,init)
	states_L= run_simulation_learned(timesteps,N,L,Nets,init)




	#plot_simulation(states,L)
	#plot_simulation(states_L,L)

	plot_simulation2(states,states_L,L)

	timeGraph(states,states_L,L)


	print('ok')