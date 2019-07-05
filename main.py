####################
# Main file, run the simulation, train the net 
# and confront the optimal with learned results
####################

from simulator1 import Simulator
import numpy as np
from no_com_net import Net
from com_net import ComNet
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
    net_inputs = []
    net_outputs = []

    # Run Simulations
    sim = Simulator(timesteps,N,L)
    for i in range(n_simulation):
        states, target_vels = sim.run()

        net_inputs.append(states)
        net_outputs.append(target_vels)
    
    # Training the net
#   nn = Net(input_size=2, epochs=10)
#    nn = CNet(input_size=2, epochs=100)
    nn = ComNet(N, 2, 2, 1, epochs = 100)
    X, Y  =nn.prepare_data(net_inputs,net_outputs)
    history = nn.train(X, Y, learning_rate=0.1)
    print('Training end')

    # Make a confront of the two simulations
    init = []
    for i in range(N):
        init.append(np.matmul(np.eye(3), mktr(rand.random()*L, 0)))


    states, _ = sim.run(init)
    states_L= sim.run_learned(nn,init, com = 1)
    nn.sess.close()

    plot_simulation2(states,states_L,L)
    timeGraph(states,states_L,L)

    print('terminated')