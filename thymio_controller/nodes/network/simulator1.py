####################
# Simulator for task 1
####################

## il clippin gdella velocità andrebbe fatto dopo aer salvato la velocità target e dopo aver salvato lo stato nel learning


from holonomic_agent import Agent, mktr, mkrot, atr
import numpy as np
import random as rand
from network import Controller
from dataset import Trace

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

class Run:

    def __init__(self, L,controller: Controller, dt: float = 0.1):
        self.controller = controller
        self.dt = dt
        self.L = L

    def __call__(self, state_c, agents_list, goal_list,  mas_vel, state_size,epsilon: float = 0.01, T: float = np.inf
                 ) -> Trace:
        t = 0.0
        dt = 0.1
        steps: List[Trace] = []
        L = self.L
        e = 10
 #       while (e > epsilon and t < T) or t == 0:
        while (t < T) or t == 0:            
            state = state_c[:,0].tolist()      
            sensing = state_c[:,3:].tolist()      
            control, *communication = self.controller(state, sensing)
            if communication:
                communication = communication[0]
            
            e = np.mean(abs(np.array(state)-np.array(goal_list)))
            
          
            # update
            for i,v in enumerate(control):
                v = np.clip(v, -mas_vel, +mas_vel)
                agents_list[i].step(v,dt)
                control[i]=v

                #check collisions
                if not agents_list[i].check_collisions(agents_list[:i]+agents_list[i+1:],L):
                    agents_list[i].setx(state[i])
                    agents_list[i].velocity= 0.0
                    control[i]=0.0
                    
            for i in range( 0, len(agents_list)):
                agents_list[i].state, agents_list[i].vels= agents_list[i].observe(agents_list,L, i)

            steps.append(Trace(t, state, communication, sensing, control, e))  
            state_c = save_state(agents_list,state_size)
            t += 1
        return Trace(*[np.array(x) for x in zip(*steps)]) 


class Simulator:
    def __init__(self,timesteps,N,L, mas_vel):
        self.timesteps = timesteps
        self.N = N
        self.L = L
        self.mas_vel = mas_vel
        self.dt =0.1

    def run(self, init= None, control=None, parameter = None):
        n_agents = self.N
        L = self.L
        timesteps = self.timesteps
        dt= self.dt

        agents_list = []

        # Create list of agents

        if init is None:
            for i in range(n_agents):
                self.initialize_agent(agents_list)
        else:
            for i in range(n_agents):
                agents_list.append(Agent(init[i],i))

        # li ordino in base alla posizione
        agents_list.sort(key=lambda x: x.getxy()[0], reverse=False)

        # initialize agents and simulation state
        for i in range(n_agents): 
            agents_list[i].state, agents_list[i].vels = agents_list[i].observe(agents_list,L,i)

        # Initialize targets
        goal_list = [None]*n_agents
        dist = L/(n_agents+1)
        for i in range(n_agents):
            goal_list[i]=dist*(i+1)
        

        # Parameters
        if parameter == 'add_vel':
            state_size = len(agents_list[i].state)+len(agents_list[i].vels)+3
        else:
            state_size = len(agents_list[i].state)+3 # x, y, theta + stato
        states = np.zeros((timesteps,n_agents,state_size)) 
        target_vels = np.zeros((timesteps, n_agents))
        errors = np.zeros((timesteps,), dtype=np.float32)
       
        #save initial state
        states[0]= save_state(agents_list, state_size)
        #state_error = states[0,:,0]    
        #errors[0]= np.mean(abs(state_error-np.array(goal_list)))        
   
        # initialize controller
        if control is None:
            controller = PID(-5,0,0) #(-5, 0, 0)
            comms = None
            for t in range (0, timesteps):
                for i in range( 0, len(agents_list)):
                    agents_list[i].state,  agents_list[i].vels= agents_list[i].observe(agents_list,L, i)
                    error = agents_list[i].getxy()[0]-goal_list[i]
                    v = controller.step(error, dt)
                    v = np.clip(v, -self.mas_vel, +self.mas_vel)
                    target_vels[t,i] = v
                # save state
                states[t] = save_state(agents_list, state_size)
                state_error = states[t,:,0]    
                errors[t] = np.mean(abs(state_error-np.array(goal_list)))

                if agents_list[0].getxy()[0]<=0: #debug
                    import pdb; pdb.set_trace()  # breakpoint 8f194059 //


                # update agents
                for i,agent in enumerate(agents_list):
                    old_position = agent.getxy()[0]
                    agent.step(target_vels[t,i],dt)

                    #check collisions
                    if not agent.check_collisions(agents_list[:i]+agents_list[i+1:],L):
                        agent.setx(old_position)
                        target_vels[t,i]=0.0
                    


#################################################################
        else:
            net_controller = control.controller()
            net_run = Run(L, controller=net_controller, dt=0.1)
            trace = net_run( states[0], agents_list,goal_list,self.mas_vel,state_size,epsilon=0.01, T=timesteps)
            
            states[:trace.state.shape[0],:,0]=trace.state
            states[:trace.sensing.shape[0],:,3:]=trace.sensing
            target_vels[:trace.control.shape[0]]=trace.control
            errors[:trace.error.shape[0]]=trace.error
            comms= trace.communication

#################################################################
        return states, target_vels, errors, comms

    def initialize_agent(self, agent_list):
        while True:
            new_agent=Agent(np.matmul(np.eye(3), mktr(rand.random()*self.L,0)),len(agent_list))
            if new_agent.check_collisions(agent_list, self.L):
                agent_list.append(new_agent)
                break

def save_state(agents_list, state_size):
    n_agents=len(agents_list)
    state = np.zeros((n_agents, state_size))
    for i in range(n_agents):
        state[i,0]= agents_list[i].getxy()[0]
        state[i,1]= agents_list[i].getxy()[1]
        state[i,2]= agents_list[i].gettheta()
        for j in range(state_size-3):            
            state[i,3+j]= np.concatenate((agents_list[i].state,agents_list[i].vels))[j]
    return state

  