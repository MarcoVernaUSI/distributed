####################
# Main file, run the simulation, train the net 
# and confront the optimal with learned results
####################

import sys
from simulator1 import Simulator2, create_init
import numpy as np
from holonomic_agent import Agent, mktr, mkrot, atr
import random as rand
from plotter import plot_simulation2, timeGraph, plot_simulationN, timeGraphN,ComGraphL, error_plot, ComGraph, timeGraphL, timeGraphL2
from dataset import create_dataset
from network import CentralizedNetL, train_net, DistributedNetL
from com_network import ComNetL, Sync, ComNetLnoSensing
import torch


## colori tra 0 e 1# 1-p p 0 
## vedere poi comunicazione valori in riferiento a questi.
## vedere se cambio comunicazione iniziale a 0.   # FATTO

# contare quanti step ci mette a stabilizzarsi con diversi numeri di robot.

# coloro sotto come la comunicazione e sopra come control. trasformo tra 0 e 8 e coloro il numero di led corrispondente. 32- 0 valori (sopra)
# numero di step che ci mettono a stabilizzarsi
# testare con 3 - 5 robot.   testare con 6, addestrare con 6 e aumentare il numero.
# distanze uniformi e tolgo sensing.   # FATTO

# backfile quanti messaggi si perdono


# modello implementazione e risultati.

#introduzine
#related work
#modello
#implementazione
#risultati.

def save_log(N, states_dist, control_dist,states_comm, control_comm, c_com):

    comm_data = np.append(np.zeros((c_com.shape[0],1)), c_com, axis=1)
    comm_data = np.append(comm_data, np.zeros((c_com.shape[0],1)), axis=1)
    
    for i in range(N):

        sensing_dist= states_dist[:,i,3:5]
        sensing_comm= states_comm[:,i,3:5]        

        np.savetxt("logs/distr/thymio_"+str(i+1)+"_sensing.csv", sensing_dist, delimiter=",", fmt='%1.6f')
        np.savetxt("logs/distr/thymio_"+str(i+1)+"_control.csv", control_dist[:,i], delimiter=",",fmt='%1.6f')
        np.savetxt("logs/comm/thymio_"+str(i+1)+"_sensing.csv", sensing_comm, delimiter=",",fmt='%1.6f')
        np.savetxt("logs/comm/thymio_"+str(i+1)+"_control.csv", control_comm[:,i], delimiter=",",fmt='%1.6f')
        np.savetxt("logs/comm/thymio_"+str(i+1)+"_comm_tx.csv", c_com[:,i], delimiter=",",fmt='%1.6f')
        np.savetxt("logs/comm/thymio_"+str(i+1)+"_comm_rx.csv", np.stack([comm_data[:,i],comm_data[:,i+2]], axis=1), delimiter=",",fmt='%1.6f')


if __name__ == '__main__':
    # Parameters
    train = True
    n_simulation = 1000 #5000
    timesteps =10 #each timestep is 1/10 of second
    n_plots = 10 # number of plots visualized
    n_test = 100 # number of example in the test set
    comm_size= 1
    uniform_init = True

    #### Simulation parameters
    L = 1 # 1 # Distance between walls
    N = 4 # 4 # Number of agents
    mas_vel = 0.07#0.07  # capped at 0.14 m/s

    #####
    command_cnt=False if sys.argv[1]=='load' else True
    command_dis=False if sys.argv[2]=='load' else True
    command_com=False if sys.argv[3]=='load' else True
    save_cmd = True if sys.argv[4]=='save' else False
    
    # Run Simulations

    sim = Simulator2(timesteps,N,L,mas_vel, uniform = uniform_init)  # if define a min_range and a max_range ti resemble the real situation.
                                                    # For the optimal simulator we assume the sensing is illimited 
    training_set = create_dataset(sim, n_simulation,"st")
    test_set = create_dataset(sim, n_simulation//5,"st")

    # Training the Centralized Net
    # lr 0.001 e 500 epochs e batch size 100 per variable range
    if command_cnt:
        net = CentralizedNetL(N)
        training_loss, testing_loss = [], []
        train_net(epochs=100, net=net, train_dataset=training_set, test_dataset=test_set, batch_size=100, learning_rate=0.0005,
              training_loss=training_loss, testing_loss=testing_loss, criterion_= "bin");
        if save_cmd:
            torch.save(net, 'models/Centralized')
    else:
        net = torch.load('models/Centralized')


    #print(training_loss)

    # Training the Distributed Net

    d_training_set = create_dataset(sim, n_simulation, 'dis')
    d_test_set = create_dataset(sim, n_simulation//5, 'dis')

    if command_dis:
        d_net = DistributedNetL(2)
        d_training_loss, d_testing_loss = [], []
        # learning rate 0.001 oscilla tra 0.03 e 0.029
        train_net(epochs=100, net=d_net, train_dataset=d_training_set, test_dataset=d_test_set, batch_size=100, learning_rate=0.001,
              training_loss=d_training_loss, testing_loss=d_testing_loss, criterion_="bin");
        if save_cmd:
            torch.save(d_net, 'models/Distributed')
    else:
        d_net = torch.load('models/Distributed')


    # Training the communication net
    # batch size 10, epochs 100, lr 0.0001
                        #   500, lr 0.00005 per 6 robot sync
    c_training_set = create_dataset(sim, n_simulation, 'com', steps=20, comm_size = comm_size)
    c_test_set = create_dataset(sim, n_simulation//5, 'com', steps=20, comm_size = comm_size)

    if command_com:
    #    c_net = ComNetL(N=N, sync=Sync.sequential)  # changed to sync
        c_net = ComNetLnoSensing(N=N, sync=Sync.sequential)  # changed to sync
    #    c_net = ComNetLnoSensing(N=N, sync=Sync.sync)  # changed to sync
  
    #    c_net = ComNetLnoSensing(N=N, sync=Sync.sync)  # changed to sync
      
        c_training_loss, c_testing_loss = [], []
        train_net(epochs=100, net=c_net, train_dataset=c_training_set, test_dataset=c_test_set, batch_size=10, learning_rate=0.0001,
            training_loss=c_training_loss, testing_loss=c_testing_loss, criterion_="bin");

        if save_cmd:
           torch.save(c_net, 'models/Communication')
    else:
        c_net = torch.load('models/Communication')


    #print(c_training_loss)

    # Make a confront of the simulations

    test_optimal=[]
    test_cent=[]
    test_dist=[]
    test_comm=[]



    for i in range(n_plots):

        L_tmp= sim.define_L()

        init = create_init(N,L_tmp, 0.06)

        states, colors, _, _ = sim.run(init=init, Linit=L_tmp)
        statesC, colorsC, _, _ = sim.run(init=init, control = net, Linit=L_tmp)   
        statesD, colorsD, _, _ = sim.run(init=init, control = d_net, Linit=L_tmp)
        statesCom, colorsCom, _, comms = sim.run(init=init, control = c_net, Linit=L_tmp)


        print('Output\n',np.array(colorsCom))
        print('Comm\n',np.array(comms))



        test_optimal.append(colors[0])
        test_cent.append(colorsC[0])
        test_dist.append(colorsD[0])
        test_comm.append(colorsCom[0])
#        plot_simulation2(states,statesC,L_tmp, 'Centralized')
#        timeGraph(states,statesC,L_tmp, 'Centralized',['Optimal', 'Learned'])

#        plot_simulation2(states,statesD,L_tmp, 'Distributed')
#        timeGraph(states,statesD,L_tmp, 'Distributed',['Optimal', 'Learned'])

#        plot_simulationL(statesCom,colorsCom, L_tmp, 'Communication')
   

        timeGraphL2(statesCom,colorsCom,L_tmp, 'Output')

        ComGraph(states,statesCom,L_tmp, 'Communication', com=comms)
        
        ComGraphL(states,statesCom,L_tmp, 'Communication', com=comms)


    print('Optimal\n',np.array(test_optimal))
    print('Centralized\n',np.array(test_cent))
    print('Distributed\n',np.array(test_dist))
    print('Communication\n',np.array(test_comm))
    # Testing

    # create test set
    inits=[]
    lengths=[]
    for i in range(n_test):
        L_tmp= sim.define_L()
        init = create_init(N,L_tmp, 0.06)
        inits.append(init)
        lengths.append(L_tmp)

    errors = np.zeros((timesteps,), dtype= np.float32)
    error_optimal=np.zeros((n_test,timesteps), dtype= np.float32)
    error_cent=np.zeros((n_test,timesteps), dtype= np.float32)
    error_dist=np.zeros((n_test,timesteps), dtype= np.float32)
    error_comm=np.zeros((n_test,timesteps), dtype= np.float32)
    
    for idx, t_sim in enumerate(inits):
        L_tmp= lengths[idx]
        st, _, e, _ = sim.run(init=t_sim, Linit=L_tmp)
        sc,_,e_cent,_ = sim.run(init=t_sim, control= net, Linit=L_tmp)
        sd,d_control, e_dist, _ = sim.run(init=t_sim, control= d_net, Linit=L_tmp)
        scom,c_control, e_comm, c_com = sim.run(init=t_sim, control= c_net, Linit=L_tmp)


        error_optimal[idx] = error_optimal[idx] + e
        error_cent[idx] = error_cent[idx] + e_cent
        error_dist[idx] = error_dist[idx] + e_dist
        error_comm[idx] = error_comm[idx] + e_comm

#        if error_cent[idx][timesteps-1]>=1:
#            plot_simulation2(st,sc,L_tmp, 'error cent')
#            timeGraph(st,sc,v, 'error cent')

#        if n_test==1:
#            save_log(N,sd,d_control,scom,c_control,c_com)

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

    error_plot([optimal,cent,dist ,comm], ['Optimal','Centralized','Distributed','Communication'])    

    print('terminated')


