####################
# Simulator for task 1
####################

from holonomic_agent import Agent, mktr, mkrot, atr
import numpy as np
import random as rand

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

class Simulator:
    def __init__(self,timesteps,N,L):
        self.timesteps = timesteps
        self.N = N
        self.L = L

    def run(self, init= None):
        timesteps = self.timesteps
        n_agents = self.N
        L = self.L
        
        states = np.zeros((timesteps,n_agents,5)) 
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

        controller = PID(-3,0,0) #(-5, 0, 0)

        #Parameters
        mas_vel = 2 # m/s

        errors = np.zeros((timesteps,), dtype=np.float32)
        for t in range (1, timesteps):
            tot_error = 0
            for i in range( 0, len(agents_list)):
                error = agents_list[i].getxy()[0]-goal_list[i]
                tot_error = tot_error + abs(error)
                v = controller.step(error, dt)
                v = np.clip(v, -mas_vel, +mas_vel)
                agents_list[i].step(v,dt)
                target_vels[t,i] = v
            for i in range( 0, len(agents_list)):
                agents_list[i].state, agents_list[i].state_d = agents_list[i].observe(agents_list,L,dt)
            states[t] = save_state(agents_list)
            errors[t] = tot_error

        return states, target_vels, errors


    def run_learned(self, Net, input_size, init=None, com = 0):
        timesteps = self.timesteps 
        n_agents = self.N
        L = self.L
        
        states = np.zeros((timesteps,n_agents,5))
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


        # goal serve per calcolare errore, non è utilizzato dal controller
        goal_list = [None]*n_agents
        dist = L/(n_agents+1)
        for i in range(n_agents):
            goal_list[i]=dist*(i+1)



        #Parameters
        mas_vel = 1 # m/s
        if com:
            communication = np.zeros((5,2), dtype=np.float32)

        errors = np.zeros((timesteps,), dtype=np.float32)
        for t in range (1, timesteps):
            inputs = states[t-1,:,3:5]
            inputs = inputs.reshape(-1,input_size)

            tot_error = 0
            if com:
                predicted_velocities, communication = Net.predict(inputs, communication)
                for i in range( 0, len(agents_list)):
                    v = predicted_velocities[i,0]
                    v = np.clip(v, -mas_vel, +mas_vel)
                    error = agents_list[i].getxy()[0]-goal_list[i]
                    tot_error = tot_error + abs(error)
                    agents_list[i].step(v,dt)
            else:
                predicted_velocities = Net.predict(inputs)
                predicted_velocities= predicted_velocities.reshape(n_agents,-1)
                for i in range( 0, len(agents_list)):   
                    v = predicted_velocities[i,0]
                    v = np.clip(v, -mas_vel, +mas_vel)
                    error = agents_list[i].getxy()[0]-goal_list[i]
                    tot_error = tot_error + abs(error)
                    agents_list[i].step(v,dt)

            for i in range( 0, len(agents_list)):
                agents_list[i].state, agents_list[i].state_d = agents_list[i].observe(agents_list,L, dt)
            states[t] = save_state(agents_list)
            errors[t] = tot_error

        return states, errors


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

  