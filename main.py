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
import tensorflow as tf



############ validation - e poi learning rate
# aumento batch size
# diminuisco sequence length
# tutti i grafici





if __name__ == '__main__':
    # Parameters
    train = False
    n_simulation = 100
    L = 10 # Distance between walls
    N = 5 # Number of agents
    timesteps = 80 #timesteps

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
#    nn = CNet(input_size=2, epochs=100) epochs 100, batch size = 10 - con queste provo con il layer in più

    nn = ComNet(N, 2, 2, 1, epochs = 100, batch_size = 20,sequence_length = 100, n_simulation = n_simulation)
    if train:
        X, Y  =nn.prepare_data(net_inputs,net_outputs)
        history, vals = nn.train(X, Y, learning_rate=0.002) #0.08 #0.01 # prima cosa capisco il learning rate migliore
        print('Training end')
    else:
        nn.saver.restore(nn.sess, "model/model.ckpt")
        print("Model restored")

                                                            # se c'è seondo hidden layer o united riduco learning rate a 0.008
    

    # Make a confront of the two simulations
    

    for i in range(5):
        init = []
        for i in range(N):
            init.append(np.matmul(np.eye(3), mktr(rand.random()*L, 0)))

        states, _ = sim.run(init)
        states_L= sim.run_learned(nn,init, com = 1)
        
        plot_simulation2(states,states_L,L)
        timeGraph(states,states_L,L)
    nn.sess.close()

    print('terminated')