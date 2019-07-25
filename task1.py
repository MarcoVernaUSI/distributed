####################
# Main file, run the simulation, train the net 
# and confront the optimal with learned results
####################


#### Problema che vanno fuori dal muro

from simulator1 import Simulator
import numpy as np
from holonomic_agent import Agent, mktr, mkrot, atr
import random as rand
from plotter import plot_simulation2, timeGraph, plot_simulationN, timeGraphN, error_plot
from dataset import create_dataset
from network import CentralizedNet, train_net, DistributedNet
from com_network import ComNet, Sync
import torch

def create_init(N,L):
        init = []
        tmp_agent_list = []
        for j in range(N):
            while True:
                new_agent=Agent(np.matmul(np.eye(3), mktr(rand.random()*L,0)),len(tmp_agent_list))
                if new_agent.check_collisions(tmp_agent_list, L):
                    tmp_agent_list.append(new_agent)
                    init.append(new_agent.pose)
                    break
        del tmp_agent_list
        return init

if __name__ == '__main__':
    # Parameters
    train = True
    n_simulation = 1000
    L = 5 # Distance between walls
    N = 4 # Number of agents
    timesteps = 100 #timesteps
    n_plots = 2
    n_test = 100
    
    # Run Simulations
    sim = Simulator(timesteps,N,L)
    training_set = create_dataset(sim, n_simulation)
    test_set = create_dataset(sim, n_simulation//5)

    # Training the Centralized Net


#    net = CentralizedNet(N)
#    training_loss, testing_loss = [], []
#    train_net(epochs=500, net=net, train_dataset=training_set, test_dataset=test_set, batch_size=100, 
#          training_loss=training_loss, testing_loss=testing_loss);
#    torch.save(net, 'models/Centralized')

    net = torch.load('models/Centralized')

    # Training the Distributed Net

    d_training_set = create_dataset(sim, n_simulation, 'dis')
    d_test_set = create_dataset(sim, n_simulation//5, 'dis')

#    d_net = DistributedNet()
#    d_training_loss, d_testing_loss = [], []
#    train_net(epochs=100, net=d_net, train_dataset=d_training_set, test_dataset=d_test_set, batch_size=100, 
#          training_loss=d_training_loss, testing_loss=d_testing_loss);
#    torch.save(d_net, 'models/Distributed')

    d_net = torch.load('models/Distributed')

    # Training the communication net

    c_training_set = create_dataset(sim, n_simulation, 'com', steps=2)
    c_test_set = create_dataset(sim, n_simulation//5, 'com', steps=2)

    c_net = ComNet(N=N, sync=Sync.sequential)  # changed to sync
    c_training_loss, c_testing_loss = [], []

    train_net(epochs=500, net=c_net, train_dataset=c_training_set, test_dataset=c_test_set, batch_size=100, 
          training_loss=c_training_loss, testing_loss=c_testing_loss);
    torch.save(c_net, 'models/Communication')

#    c_net = torch.load('models/Communication')

    # Make a confront of the simulations
    for i in range(n_plots):
        init = create_init(N,L)

        states, _, _ = sim.run(init=init)
        statesC, _, _ = sim.run(init=init, control = net)
        statesD, _, _ = sim.run(init=init, control = d_net)
        statesCom, _, _ = sim.run(init=init, control = c_net)
     
        plot_simulation2(states,statesC,L, 'Centralized')
        timeGraph(states,statesC,L, 'Centralized')

        plot_simulation2(states,statesD,L, 'Distributed')
        timeGraph(states,statesD,L, 'Distributed')

        plot_simulation2(states,statesCom,L, 'Communication')
        timeGraph(states,statesCom,L, 'Communication')
   
    # Testing

    # create test set
    inits=[]
    for i in range(n_test):
        init = create_init(N,L)
        inits.append(init)

    errors = np.zeros((timesteps,), dtype= np.float32)
    error_optimal=np.zeros((n_test,timesteps), dtype= np.float32)
    error_cent=np.zeros((n_test,timesteps), dtype= np.float32)
    error_dist=np.zeros((n_test,timesteps), dtype= np.float32)
    error_comm=np.zeros((n_test,timesteps), dtype= np.float32)
    
    for idx, t_sim in enumerate(inits):
        st, _, e = sim.run(t_sim)
        sc,_, e_cent = sim.run(init=t_sim, control= net)
        sd,_, e_dist = sim.run(init=t_sim, control= d_net)
        sd,_, e_comm = sim.run(init=t_sim, control= c_net)

        error_optimal[idx] = error_optimal[idx] + e
        error_cent[idx] = error_cent[idx] + e_cent
        error_dist[idx] = error_dist[idx] + e_dist
        error_comm[idx] = error_comm[idx] + e_comm

        if error_cent[idx][timesteps-1]>=1:
            plot_simulation2(st,sc,L, 'error cent')
            timeGraph(st,sc,L, 'error cent')

    max_optimal=np.quantile(error_optimal,0.95 ,axis=0)
    min_optimal=np.quantile(error_optimal,0.05 ,axis=0)
    error_optimal=np.mean(error_optimal, axis=0)
    optimal=np.stack((error_optimal, max_optimal, min_optimal),axis=1)

    max_cent=np.quantile(error_cent,0.95 ,axis=0)
    min_cent=np.quantile(error_cent,0.05 ,axis=0)
    error_cent=np.mean(error_cent, axis=0)
    cent=np.stack((error_cent, max_cent, min_cent),axis=1)

    max_dist=np.quantile(error_dist,0.95 ,axis=0)
    min_dist=np.quantile(error_dist,0.05 ,axis=0)
    error_dist=np.mean(error_dist, axis=0)
    dist=np.stack((error_dist, max_dist, min_dist),axis=1)


    max_comm=np.quantile(error_comm,0.95 ,axis=0)
    min_comm=np.quantile(error_comm,0.05 ,axis=0)
    error_comm=np.mean(error_comm, axis=0)
    comm=np.stack((error_comm, max_comm, min_comm),axis=1)

    error_plot([optimal,cent,dist, comm])    

    print('terminated')