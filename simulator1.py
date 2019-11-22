####################
# Simulator for task 1
####################

from holonomic_agent import Agent, mktr, mkrot, atr
import numpy as np
import random as rand
from network import Controller
from dataset import Trace
import sys

def Cross_Entropy(y_hat, y):
    if y == 1:
      return -np.log(y_hat)
    else:
      return -np.log(1 - y_hat)

# Function to generate the random samples
def ranks(sample):
    """
    Return the ranks of each element in an integer sample.
    """
    indices = sorted(range(len(sample)), key=lambda i: sample[i])
    return sorted(indices, key=lambda i: indices[i])

def sample_with_minimum_distance(n=100, k=5, d=13):
    """
    Sample of k elements from range(n), with a minimum distance d.
    """
    sample = rand.sample(range(n-(k-1)*(d-1)), k)
    return [s + (d-1)*r for s, r in zip(sample, ranks(sample))]

def create_init_old(N,L):
        init = []
        tmp_agent_list = []
        centroid_d = L/(N+1)

        for j in range(N):
            centroid = centroid_d*(j+1)
            while True:
                
                new_agent=Agent(np.matmul(np.eye(3), mktr(centroid + [-1,1][rand.randrange(2)]*(rand.random()*(centroid_d/2-0.06)),0)),len(tmp_agent_list))  # 0.3 è metà dalla width di un robot
    
                if new_agent.check_collisions(tmp_agent_list, L):
                    tmp_agent_list.append(new_agent)
                    init.append(new_agent.pose)
                    break
        del tmp_agent_list
        return init

def create_init(N,L, agent_width, buffer_distance = 0.01):

        init = []
        tmp_agent_list = []

        # create initial positions
        offset = int(round((agent_width + buffer_distance)*100))
        range_ = int(round(L*100)) - 2*offset # lo trasformo in centimetri
          # qui controllare che non faccia casini, nel caso da errore aumntare la buffer_distance
        
        initial_positions = np.array(sample_with_minimum_distance(range_,N,offset*2))        
        initial_positions = initial_positions + offset  # sommo a tutti l'offset
        # divido per cento per tornare a metri
        initial_positions = initial_positions / 100

        for j in range(N):
            while True:
                new_agent=Agent(np.matmul(np.eye(3), mktr(initial_positions[j],0)),len(tmp_agent_list))  # 0.3 è metà dalla width di un robot
                if new_agent.check_collisions(tmp_agent_list, L):
                    tmp_agent_list.append(new_agent)
                    init.append(new_agent.pose)
                    break
                    # assert che non ci siano collisioni
                else:
                    print('inizializzazione agenti fallita')
                    sys.exit()

        del tmp_agent_list
        return init

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

class RunL:

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
            
            e = []
            for i in range(len(control)):
                e.append(Cross_Entropy(control[i], goal_list[i]))

            e = np.mean(np.array(e))                    
            for i in range( 0, len(agents_list)):
                agents_list[i].state, agents_list[i].vels= agents_list[i].observe(agents_list,L, i)

            steps.append(Trace(t, state, communication, sensing, control, e))  
            state_c = save_state(agents_list,state_size)
            t += 1
        return Trace(*[np.array(x) for x in zip(*steps)]) 


class Simulator:
    def __init__(self,timesteps,N,L, mas_vel, min_range = 0.0, max_range = 9999999.0, uniform = False):
        self.timesteps = timesteps
        self.N = N
        self.L = L
        self.mas_vel = mas_vel
        self.dt =0.1
        self.min_range = min_range
        self.max_range = max_range
        self.uniform = uniform

    def define_L(self):
        L = self.L
        return L

    def define_N(self):
        N = self.N
        return N

    def run(self, init= None, control=None, parameter = None, goal_set = True, Linit=None):
        n_agents = self.define_N()
        timesteps = self.timesteps
        dt= self.dt
        L = self.define_L()

        agents_list = []

        # Create list of agents
        if self.uniform:
            init=create_init_uniform(n_agents,L,0.06)
        else:
            if init is None:            
                init=create_init(n_agents,L,0.06)
            else:
                L = Linit            
        for i in range(n_agents):                
            agents_list.append(Agent(init[i],i, self.min_range, self.max_range))

        # li ordino in base alla posizione
        agents_list.sort(key=lambda x: x.getxy()[0], reverse=False)

        # initialize agents and simulation state

        for i in range(n_agents): 
            agents_list[i].state, agents_list[i].vels = agents_list[i].observe(agents_list,L,i)

        # Initialize targets
        goal_list = [None]*n_agents
        

        if goal_set:
            agent_width = agents_list[0].width
            free_space = L - (n_agents*agent_width*2)
            # assert free_space > 0
            space_dist = free_space / (n_agents + 1)
            # first agent
            goal_list[0]= space_dist + agent_width
            if n_agents > 2:
                for i in range(1,n_agents-1):
                    goal_list[i]= space_dist + agent_width + (2*agent_width + space_dist)*i            
            if n_agents > 1:
                goal_list[n_agents-1]=L - (agent_width + space_dist)
        else:
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


    #def initialize_agent(self, agent_list, centroid, i):
    #    while True:
    #        new_agent=Agent(np.matmul(np.eye(3), mktr(centroid*(i+1) + [-1,1][rand.randrange(2)]*(rand.random()*(centroid/2-0.06)),0)),len(agent_list), self.min_range, self.max_range)  # 0.3 è metà dalla width di un robot
    #        #new_agent=Agent(np.matmul(np.eye(3), mktr(rand.random()*(self.L),0)),len(agent_list), self.min_range, self.max_range)  # 0.3 è metà dalla width di un robot
    #        if new_agent.check_collisions(agent_list, self.L):
    #            agent_list.append(new_agent)
    #            break


class SimulatorR(Simulator):
    def __init__(self,timesteps,N,mL,ML,mas_vel, min_range = 0.0, max_range = 9999999.0, uniform = False):
        self.timesteps = timesteps
        self.N = N
        self.mL = mL
        self.ML = ML
        self.mas_vel = mas_vel
        self.dt =0.1
        self.min_range = min_range
        self.max_range = max_range
        self.uniform = uniform

    def define_L(self):
        L = round(random.uniform(self.mL,self.ML), 2)
        return L

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

class Simulator2(Simulator):
    def __init__(self,timesteps,N,L, mas_vel, min_range = 0.0, max_range = 9999999.0, uniform = False):
        super(Simulator2, self).__init__(timesteps,N,L, mas_vel, min_range, max_range, uniform)

    def run(self, init= None, control=None, parameter = None, Linit=None):
        n_agents = self.define_N()
        
        if init is not None:
            if len(init)!=n_agents:
                n_agents=len(init)

        timesteps = self.timesteps
        dt= self.dt
        L = self.define_L()

        agents_list = []

        # Create list of agents
        if self.uniform:
            init=create_init_uniform(n_agents,L,0.06)
        else:
            if init is None:            
                init=create_init(n_agents,L,0.06)
            else:                
                L = Linit            
        for i in range(n_agents):                
            agents_list.append(Agent(init[i],i, self.min_range, self.max_range))

        # li ordino in base alla posizione
        agents_list.sort(key=lambda x: x.getxy()[0], reverse=False)

        # initialize agents and simulation state

        for i in range(n_agents): 
            agents_list[i].state, agents_list[i].vels = agents_list[i].observe(agents_list,L,i)

        # Initialize targets
        goal_list = [None]*n_agents
        for i in range(n_agents//2):
            goal_list[i]=0
        i=i+1
        for j in range(len(goal_list)-i):
            goal_list[i+j]=1        

        # Parameters
        state_size = len(agents_list[i].state)+3 # x, y, theta + stato
        states = np.zeros((timesteps,n_agents,state_size)) 
        target_colors = np.zeros((timesteps, n_agents))
        errors = np.zeros((timesteps,), dtype=np.float32)
       
        #save initial state
        states[0]= save_state(agents_list, state_size)
   
        # initialize controller
        if control is None:
            comms = None
            for t in range (0, timesteps):
                for i in range( 0, len(agents_list)):
                    agents_list[i].state,  agents_list[i].vels= agents_list[i].observe(agents_list,L, i)
                    agents_list[i].color = goal_list[i]
                    target_colors[t,i] = goal_list[i]
                # save state

                states[t] = save_state(agents_list, state_size)                
                errors[t] = 0.0

#################################################################
        ### ATTENZIONE CHE TEST SYNC LI O FATTI CON VECCHIA VERSIONE IN CUI IL NET CONTROLLER ERA SEMPRE SEQUENZIALE
        ### Domanda funiona megli sync - sync.  sequenziale - sync. oppure sync - sequenziale?
        else:
            net_controller = control.controller()            
            net_run = RunL(L, controller=net_controller, dt=0.1)
            trace = net_run( states[0], agents_list,goal_list,self.mas_vel,state_size,epsilon=0.01, T=timesteps)            
            states[:trace.state.shape[0],:,0]=trace.state
            states[:trace.sensing.shape[0],:,3:]=trace.sensing
            target_colors[:trace.control.shape[0]]=trace.control
            errors[:trace.error.shape[0]]=trace.error
            comms= trace.communication

#################################################################
        return states, target_colors, errors, comms

def create_init_uniform(N,L, agent_width):

        init = []
        tmp_agent_list = []

        # create initial positions

        distance_between_agents= (L - ((agent_width*2)*N))/ (N+1)
        
        initial_positions = []
        j = 1
        for i in range(N):
            x = i+1
            initial_positions.append(x*distance_between_agents + j*agent_width)
            j = j+2

        for j in range(N):
            while True:
                new_agent=Agent(np.matmul(np.eye(3), mktr(initial_positions[j],0)),len(tmp_agent_list))  # 0.3 è metà dalla width di un robot
                if new_agent.check_collisions(tmp_agent_list, L):
                    tmp_agent_list.append(new_agent)
                    init.append(new_agent.pose)
                    break
                    # assert che non ci siano collisioni
                else:
                    print('inizializzazione agenti fallita')
                    sys.exit()

        del tmp_agent_list
        return init


class SimulatorN(Simulator2):
    def __init__(self,timesteps,Ns,L, mas_vel, min_range = 0.0, max_range = 9999999.0, uniform = False):
        self.timesteps = timesteps
        self.Ns = Ns
        self.L = L
        self.mas_vel = mas_vel
        self.dt =0.1
        self.min_range = min_range
        self.max_range = max_range
        self.uniform = uniform

    def define_N(self):
        N = rand.choice(self.Ns)
        return N
