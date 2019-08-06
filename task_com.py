####################
# Main file, run the simulation, train the net 
# and confront the optimal with learned results
####################


#### Problema che vanno fuori dal muro
import sys
from simulator1 import Simulator
import numpy as np
from holonomic_agent import Agent, mktr, mkrot, atr
import random as rand
from plotter import plot_simulation2, timeGraph, plot_simulationN, timeGraphN, error_plot, ComGraph
from dataset import create_dataset
from network import CentralizedNet, train_net, DistributedNet
from com_network import ComNet, Sync, Com2Net
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
    n_test = 10
    mas_vel = 3

    #####
    command_disv=False if sys.argv[1]=='load' else True
    command_2com=False if sys.argv[2]=='load' else True
    save_cmd = True if sys.argv[3]=='save' else False
    
    # Run Simulations
    sim = Simulator(timesteps,N,L,mas_vel)

    # Training the Distributed velocity Net

    dv_training_set = create_dataset(sim, n_simulation, 'dis',param2='add_vel')
    dv_test_set = create_dataset(sim, n_simulation//5, 'dis',param2 = 'add_vel')

    if command_disv:
        dv_net = DistributedNet(4)
        dv_training_loss, dv_testing_loss = [], []
        train_net(epochs=800, net=dv_net, train_dataset=dv_training_set, test_dataset=dv_test_set, batch_size=100, 
              training_loss=dv_training_loss, testing_loss=dv_testing_loss, learning_rate= 0.0008);
        
        if save_cmd:
            torch.save(dv_net, 'models/DistributedV')
    else:
        dv_net = torch.load('models/DistributedV')

######################################################################
    # Training the 2-communication net

    c_training_set = create_dataset(sim, n_simulation, 'com', steps=2)
    c_test_set = create_dataset(sim, n_simulation//5, 'com', steps=2)

    if command_2com:
        c2_net = Com2Net(N=N, sync=Sync.sequential)  # changed to sync
        c_training_loss, c_testing_loss = [], []
        train_net(epochs=500, net=c2_net, train_dataset=c_training_set, test_dataset=c_test_set, batch_size=100, 
            training_loss=c_training_loss, testing_loss=c_testing_loss);
        if save_cmd:
            torch.save(c2_net, 'models/2Communication')
    else:
        c2_net = torch.load('models/2Communication')
#############################################################################################
    import pdb; pdb.set_trace()  # breakpoint ae57ecaa //


    # Load distributed and com net
    d_net = torch.load('models/Distributed')
    c_net = torch.load('models/Communication')



    # Make a confront of the simulations
    for i in range(n_plots):
        init = create_init(N,L)


        states, _, _, _ = sim.run(init=init)
        statesDV, _, _, _ = sim.run(init=init, control = dv_net, parameter= 'add_vel')
        statesD, _, _, _ = sim.run(init=init, control = d_net)
        statesCom, _, _, comms = sim.run(init=init, control = c_net)
        statesCom2, _, _, comms2 = sim.run(init=init, control = c2_net)
        

        plot_simulation2(statesD,statesDV,L, 'Distributed Confront')
        timeGraph(statesD,statesDV,L, 'Distributed Confront',['Only distance', 'Distance & velocity'])

        plot_simulation2(statesCom,statesCom2,L, 'Communication Confront')
        timeGraph(statesCom,statesCom2,L, 'Communication Confront',['Com', '2 Com'])

        plot_simulation2(states,statesCom,L, 'Communication Confront')
        timeGraph(states,statesCom,L, 'Communication Confront',['Optimal', 'Com'])

        plot_simulation2(states,statesCom2,L, 'Communication Confront')
        timeGraph(states,statesCom2,L, 'Communication Confront',['Optimal', '2 Com'])
        # Le due comm
    #   ComGraph(states,statesCom,L, 'Communication', com=comms)
        # Stampo comunicazione 2



    # Testing
    import pdb; pdb.set_trace()  # breakpoint e1b45ce1 //

    # create test set
    inits=[]
    for i in range(n_test):
        init = create_init(N,L)
        inits.append(init)

    errors = np.zeros((timesteps,), dtype= np.float32)
    error_optimal=np.zeros((n_test,timesteps), dtype= np.float32)
    error_dist=np.zeros((n_test,timesteps), dtype= np.float32)
    error_distv=np.zeros((n_test,timesteps), dtype= np.float32)
    error_comm=np.zeros((n_test,timesteps), dtype= np.float32)
    error_comm2=np.zeros((n_test,timesteps), dtype= np.float32)
    

    for idx, t_sim in enumerate(inits):
        st, _, e, _ = sim.run(t_sim)
        sd,_, e_dist, _ = sim.run(init=t_sim, control= d_net)
        sc,_, e_comm, _ = sim.run(init=t_sim, control= c_net)
        sdc,_, e_vdist, _ = sim.run(init=t_sim, control= dv_net, parameter= 'add_vel')
        sc2,_, e_2comm, _ = sim.run(init=t_sim, control= c2_net)



        error_optimal[idx] = error_optimal[idx] + e
        error_dist[idx] = error_dist[idx] + e_dist
        error_comm[idx] = error_comm[idx] + e_comm
        error_distv[idx] = error_distv[idx] + e_vdist
        error_comm2[idx] = error_comm2[idx] + e_2comm

    max_optimal=np.quantile(error_optimal,0.95 ,axis=0)
    min_optimal=np.quantile(error_optimal,0.05 ,axis=0)
    error_optimal=np.mean(error_optimal, axis=0)
    optimal=np.stack((error_optimal, max_optimal, min_optimal),axis=1)

    max_dist=np.quantile(error_dist,0.95 ,axis=0)
    min_dist=np.quantile(error_dist,0.05 ,axis=0)
    error_dist=np.mean(error_dist, axis=0)
    dist=np.stack((error_dist, max_dist, min_dist),axis=1)

    max_comm=np.quantile(error_comm,0.95 ,axis=0)
    min_comm=np.quantile(error_comm,0.05 ,axis=0)
    error_comm=np.mean(error_comm, axis=0)
    comm=np.stack((error_comm, max_comm, min_comm),axis=1)

    max_distv=np.quantile(error_distv,0.95 ,axis=0)
    min_distv=np.quantile(error_distv,0.05 ,axis=0)
    error_distv=np.mean(error_distv, axis=0)
    vdist=np.stack((error_distv, max_distv, min_distv),axis=1)

    max_comm2=np.quantile(error_comm2,0.95 ,axis=0)
    min_comm2=np.quantile(error_comm2,0.05 ,axis=0)
    error_comm2=np.mean(error_comm2, axis=0)
    comm2=np.stack((error_comm2, max_comm2, min_comm2),axis=1)


    error_plot([optimal,dist, vdist, comm, comm2], ['Optimal','Distributed','Distributed + Vel','Communication','2 Communication'])    

    print('terminated')