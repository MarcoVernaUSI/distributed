####################
# Main file, run the simulation, train the net 
# and confront the optimal with learned results
####################


from simulator1 import Simulator
import numpy as np
from no_com_net import Net
from holonomic_agent import Agent, mktr, mkrot, atr
import random as rand
from plotter import plot_simulation2, timeGraph

if __name__ == '__main__':
    # Parameters
    n_simulation = 100
    L = 10 # Distance between walls
    N = 5 # Number of agents
    timesteps = 100 #timesteps

    # prepare data for the net
    net_inputs = None
    net_outputs = None

    # Run Simulations
    sim = Simulator(timesteps,N,L)
    for i in range(n_simulation):
        states, target_vels = sim.run()
        
        # prepara data for the training of the net
        t_data=states[:,:,3:5]
        t_data = t_data.reshape(t_data.shape[0], -1)

        if net_inputs is None:
            net_inputs = t_data
            net_outputs=target_vels
        else:
            net_inputs = np.append(net_inputs,t_data,axis=0)
            net_outputs = np.append(net_outputs,target_vels,axis=0)
    # Training the net
    nn = Net(input_size=2, epochs=10)
    history = nn.train(net_inputs, net_outputs)

    # Make a confront of the two simulations
    init = []
    for i in range(N):
        init.append(np.matmul(np.eye(3), mktr(rand.random()*L, 0)))

    states, _ = sim.run(init)
    states_L= sim.run_learned(nn,init)

    plot_simulation2(states,states_L,L)
    timeGraph(states,states_L,L)

    print('terminated')