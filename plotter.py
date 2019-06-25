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

def update_plot(i, x_data1,y_data1, x_data2,y_data2, scat1, scat2):
    data1 = np.array([x_data1[i],y_data1[i]]).T
    scat1.set_offsets(data1)
    data2 = np.array([x_data2[i],y_data2[i]]).T
    scat2.set_offsets(data2)
    
    return scat1, scat2
        
