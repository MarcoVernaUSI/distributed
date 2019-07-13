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
from plotter import plot_simulation2, timeGraph, plot_simulationN, timeGraphN, error_plot
import tensorflow as tf

if __name__ == '__main__':
    # Parameters
    train = False
    n_simulation = 200
    L = 10 # Distance between walls
    N = 5 # Number of agents
    timesteps = 80 #timesteps
    n_plots = 2
    n_test = 50

    # prepare data for the net
    net_inputs = []
    net_outputs = []

    # Run Simulations
    sim = Simulator(timesteps,N,L)
    for i in range(n_simulation):
        states, target_vels, _ = sim.run()

        net_inputs.append(states)
        net_outputs.append(target_vels)
    
    # Training the Centralized net
    nn_cent = Net(input_size=10, epochs=50)
    X, Y  =nn_cent.prepare_data(net_inputs,net_outputs)
    history = nn_cent.train(X, Y) 

    # Training the distributed net
    nn_dist = Net(input_size=2, epochs=50)
    X, Y  =nn_dist.prepare_data(net_inputs,net_outputs)
    history = nn_dist.train(X, Y) 

    # Training the comminication Net
    nn = ComNet(N, 2, 2, 1, epochs = 200, batch_size = 20,sequence_length = timesteps, n_simulation = n_simulation)
    if train:
        X, Y  =nn.prepare_data(net_inputs,net_outputs)
        history, vals = nn.train(X, Y, learning_rate=0.0001) #0.08 #0.0001 # prima cosa capisco il learning rate migliore
        print('Training end')
    else:
        nn.saver.restore(nn.sess, "model/model.ckpt")
        print("Model restored")


    # Make a confront of the simulations
    
    for i in range(n_plots):
        init = []
        for i in range(N):
            init.append(np.matmul(np.eye(3), mktr(rand.random()*L, 0)))

        states, _, _ = sim.run(init)
        statesC, _ = sim.run_learned(nn_cent, nn_cent.input_size, init, com=0)
        statesD, _ = sim.run_learned(nn_dist, nn_dist.input_size, init, com=0)
        states_L, _= sim.run_learned(nn,nn.input_size,init, com = 1)
     
        plot_simulation2(states,statesC,L)
        timeGraph(states,statesC,L)

        plot_simulation2(states,statesD,L)
        timeGraph(states,statesD,L)

        plot_simulation2(states,states_L,L)
        timeGraph(states,states_L,L)

    # Testing

    # create test set
    inits=[]
    for i in range(n_test):
        init = []
        for i in range(N):
            init.append(np.matmul(np.eye(3), mktr(rand.random()*L, 0)))
        inits.append(init)


    errors = np.zeros((timesteps,), dtype= np.float32)
    error_optimal=np.zeros((n_test,timesteps), dtype= np.float32)
    error_cent=np.zeros((n_test,timesteps), dtype= np.float32)
    error_dist=np.zeros((n_test,timesteps), dtype= np.float32)
    error_com=np.zeros((n_test,timesteps), dtype= np.float32)
    
    for idx, t_sim in enumerate(inits):
        _, _, e = sim.run(t_sim)
        _, e_cent = sim.run_learned(nn_cent, nn_cent.input_size, t_sim, com=0)
        _, e_dist = sim.run_learned(nn_dist, nn_dist.input_size, t_sim, com=0)
        _, e_c = sim.run_learned(nn, nn.input_size, t_sim, com=1)
        error_optimal[idx] = error_optimal[idx] + e
        error_cent[idx] = error_cent[idx] + e_cent
        error_dist[idx] = error_dist[idx] + e_dist
        error_com[idx] = error_com[idx] + e_c

    error_optimal=np.mean(error_optimal, axis=0)
    error_cent=np.mean(error_cent, axis=0)
    error_dist=np.mean(error_dist, axis=0)
    error_com=np.mean(error_com, axis=0)

    nn.sess.close()

    error_plot([error_optimal[1:],error_cent[1:],error_dist[1:],error_com[1:]])


    print('terminated')