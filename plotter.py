####################
# Plotter
####################

from matplotlib import pyplot, transforms
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def plot_simulation(data, L):

    numframes = len(data)-1
    numpoints = len(data[0])
    x = data[0,:,0]
    y = data[0,:,1]

    x_data = data[1:,:,0]
    
    y_data = data[1:,:,1]

    fig = plt.figure()
    

    #data = np.random.rand(numframes, 2,5)
    l, = plt.plot([], [], 'r-')
    plt.xlim(0, L)
    plt.ylim(-1, 1)
    plt.xlabel('x')
    plt.title('Task 1')


    line = np.linspace(0, L)
    plt.plot(line, 0*line);

    scat = plt.scatter(x, y, s=100)
#   points = plt.plot([1,2,3,4,5], [0,0,0,0,0], 'ro')


    for i in range(numframes):
        ani = animation.FuncAnimation(fig, update_plot ,frames = numframes ,fargs=(x_data,y_data, scat))


    plt.show()

def plot_simulation2(data1,data2, L):

    numframes = len(data1)-1
    numpoints = len(data1[0])
    x1 = data1[0,:,0]
    y1 = data1[0,:,1]

    x2 = data2[0,:,0]
    y2 = data2[0,:,1]



    x_data1 = data1[1:,:,0] 
    y_data1 = data1[1:,:,1]

    x_data2 = data2[1:,:,0] 
    y_data2 = data2[1:,:,1]


    fig = plt.figure()
    

    #data = np.random.rand(numframes, 2,5)
    l, = plt.plot([], [], 'r-')
    plt.xlim(0, L)
    plt.ylim(-1, 1)
    plt.xlabel('x')
    plt.title('Task 1')


    line = np.linspace(0, L)
    plt.plot(line, 0*line);

    scat1 = plt.scatter(x1, y1, s=100)
    scat2 = plt.scatter(x2, y2, s=100, c='red')
#   points = plt.plot([1,2,3,4,5], [0,0,0,0,0], 'ro')


    for i in range(numframes):
        ani = animation.FuncAnimation(fig, update_plot ,frames = numframes ,fargs=(x_data1,y_data1,x_data2,y_data2, scat1, scat2))

    ani.save('plots/animation.gif', writer='imagemagick', fps=10)
    plt.show()


def plot_simulationN(data_all, L):



    Xs = []
    Ys = []

    Xs_data=[]
    Ys_data=[]


    for i, data in enumerate(data_all):

        numframes = len(data)-1
        numpoints = len(data[0])

        Xs.append(data[0,:,0])
        Ys.append(data[0,:,1])

        Xs_data.append(data[1:,:,0]) 
        Ys_data.append(data[1:,:,1])

    fig = plt.figure()
    
    #data = np.random.rand(numframes, 2,5)
    l, = plt.plot([], [], 'r-')
    plt.xlim(0, L)
    plt.ylim(-1, 1)
    plt.xlabel('x')
    plt.title('Task 1')

    line = np.linspace(0, L)
    plt.plot(line, 0*line);

    scats=[]
    colors = ['blue','red', 'brown', 'green']

    for i in range(len(data_all)):
        scats.append(plt.scatter(Xs[i], Ys[i], s=100, c=colors[i]))
  
    for i in range(numframes):
        ani = animation.FuncAnimation(fig, update_plotN ,frames = numframes ,fargs=(Xs_data,Ys_data,scats))

    ani.save('plots/animation.gif', writer='imagemagick', fps=10)
    plt.show()


def timeGraph(data1,data2, L):

    x_data1 = data1[0:,:,0] 
    y_data1 = data1[0:,:,1]

    x_data2 = data2[0:,:,0] 
    y_data2 = data2[0:,:,1]


    fig = plt.figure()

    base = pyplot.gca().transData
    rot = transforms.Affine2D().rotate_deg(90)

    plt.plot(x_data1,color='blue',label = 'optimal',transform= rot + base)
    plt.plot(x_data2,color='red',label = 'learned',transform= rot + base)
    
    custom_lines = [Line2D([0], [0], color='blue', lw=4),Line2D([0], [0], color='red', lw=4)]
    plt.legend(custom_lines, ['Optimal', 'Learned'],loc=4)

    plt.xlabel('x position')
    plt.ylabel('timesteps')
    plt.title('Task 1')


    plt.show()  



def timeGraphN(data_all, L):

    x_data=[]
    y_data=[]

    for i,data in enumerate(data_all):
        x_data.append(data[0:,:,0]) 
        y_data.append(data[0:,:,1])

    fig = plt.figure()

    base = pyplot.gca().transData
    rot = transforms.Affine2D().rotate_deg(90)

    plt.plot(x_data[0],color='blue',label = 'optimal',transform= rot + base)
    plt.plot(x_data[1],color='red',label = 'centralized',transform= rot + base)
    plt.plot(x_data[2],color='brown',label = 'distributed',transform= rot + base)
    plt.plot(x_data[3],color='green',label = 'learned',transform= rot + base)

    
    custom_lines = [Line2D([0], [0], color='blue', lw=4),Line2D([0], [0], color='red', lw=4),Line2D([0], [0], color='brown', lw=4),Line2D([0], [0], color='green', lw=4)]
    plt.legend(custom_lines, ['Optimal','Centralized' ,'Distributed','Learned'],loc=4)

    plt.xlabel('x position')
    plt.ylabel('timesteps')
    plt.title('Task 1')

    plt.show()  

# update per i plot da 4
def update_plotN(i, Xs_data, Ys_data, scats):
    for n in range(len(Xs_data)):
        data = np.array([Xs_data[n][i],Ys_data[n][i]]).T
        scats[n].set_offsets(data)

    return scats[0], scats[1], scats[2], scats[3]



def update_plot(i, x_data1,y_data1, x_data2,y_data2, scat1, scat2):
    data1 = np.array([x_data1[i],y_data1[i]]).T
    scat1.set_offsets(data1)
    data2 = np.array([x_data2[i],y_data2[i]]).T
    scat2.set_offsets(data2)
    
    return scat1, scat2
        

def error_plot(errors):

    timesteps = np.linspace(0, len(errors[0]), num=len(errors[0]))

    plt.grid(True, which='both')

    # Linear X axis, Logarithmic Y axis
    plt.plot(timesteps, errors[0] , label='Optimal')
    plt.plot(timesteps, errors[1] , label='Centralized')
    plt.plot(timesteps, errors[2] , label='Distributed')
    plt.plot(timesteps, errors[3] , label='Learned')
    plt.legend(loc=3)

    plt.title('Loss for every net')
    plt.xlabel('Timestep')
    plt.ylabel('Loss')

    plt.show()


        
