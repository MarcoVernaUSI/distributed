####################
# Plotter
####################

from matplotlib import pyplot, transforms
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm
import numpy as np

def plot_simulation(data, L, name):

    numframes = len(data)-1
    numpoints = len(data[0])
    x = data[0,:,0]
    y = data[0,:,1]

    x_data = data[1:,:,0]
    
    y_data = data[1:,:,1]

    fig = plt.figure()
    

    l, = plt.plot([], [], 'r-')
    plt.xlim(-0.5, L+0.5)
    plt.ylim(-1, 1)
    plt.xlabel('x')
    plt.title(name)


    line = np.linspace(0, L)
    wall = [np.linspace(-3, 3)]
    plt.plot(line, 0*line);
    plt.plot(0*wall,wall);
    plt.plot(L+0*wall, wall);

    scat = plt.scatter(x, y, s=100)

    for i in range(numframes):
        ani = animation.FuncAnimation(fig, update_plot ,frames = numframes ,fargs=(x_data,y_data, scat))


    plt.show()

def plot_simulation2(data1,data2, L,name):

    numframes = len(data1)-1
    numpoints = len(data1[0])
    x1 = data1[0,:,0]
    y1 = data1[0,:,1]

    x2 = data2[0,:,0]
    y2 = data2[0,:,1]



    x_data1 = data1[1:,:,0] 
    y_data1 = data1[1:,:,1]
    x_data1_right = x_data1 + 0.06
    x_data1_left = x_data1 - 0.06

    x_data2 = data2[1:,:,0] 
    y_data2 = data2[1:,:,1]
    x_data2_right = x_data2 + 0.06
    x_data2_left = x_data2 - 0.06


    fig = plt.figure()
    

    l, = plt.plot([], [], 'r-')
    plt.xlim(-0.5, L+0.5)
    plt.ylim(-1, 1)
    plt.xlabel('x')
    plt.title(name)


    line = np.linspace(0, L)
    wall = np.linspace(-3, 3)
    plt.plot(line, 0*line);
    plt.plot(0*wall,wall, color='black');
    plt.plot(L+0*wall, wall, color='black');



    scat1 = plt.scatter(x1, y1, s=80, c='blue')
    scat1_right = plt.scatter(x1+0.06, y1, s=30, c='blue',marker="<")
    scat1_left = plt.scatter(x1-0.06, y1, s=30, c='blue',marker=">")
        

    scat2 = plt.scatter(x2, y2, s=80, c='red')
    scat2_right = plt.scatter(x2+0.06, y1, s=30, c='red',marker="<")
    scat2_left = plt.scatter(x2-0.06, y1, s=30, c='red', marker=">")
 
    for i in range(numframes):
        ani = animation.FuncAnimation(fig, update_plot ,frames = numframes ,fargs=(x_data1,y_data1,x_data1_right,y_data1,x_data1_left,y_data1, x_data2,y_data2,x_data2_right,y_data2,x_data2_left,y_data2 ,scat1,scat1_right,scat1_left, scat2,scat2_right,scat2_left))

    ani.save('plots/animation.gif', writer='imagemagick', fps=10)
    plt.show()

def plot_simulation_task2(data1,data2, L,name):




    numframes = len(data1)-1
    numpoints = len(data1[0])
    x1 = data1[0,:,0]
    y1 = data1[0,:,1]

    x_data1 = data1[1:,:,0] 
    y_data1 = data1[1:,:,1]


    fig = plt.figure()
    

    l, = plt.plot([], [], 'r-')
    plt.xlim(-0.5, L+0.5)
    plt.ylim(-1, 1)
    plt.xlabel('x')
    plt.title(name)


    line = np.linspace(0, L)
    wall = np.linspace(-3, 3)
    plt.plot(line, 0*line);
    plt.plot(0*wall,wall, color='black');
    plt.plot(L+0*wall, wall, color='black');



    c1=data2[0]
    c_data = data2[1:]
    #split
    x1_blue=[]
    x1_red=[]
    for i in range(x1.shape[0]):
        if c1[i]<0.5:
            x1_blue.append(x1[i])
        else:
            x1_red.append(x1[i])

    x_data1_blue=[]
    x_data1_red=[]
    for j in range(x_data1.shape[0]):
        x_data1_blue_line=[]
        x_data1_red_line=[]
        for i in range(x_data1.shape[1]):
            if c_data[j,i]<0.5:
                x_data1_blue_line.append(x_data1[j,i])
            else:
                x_data1_red_line.append(x_data1[j,i])
        x_data1_blue.append(np.array(x_data1_blue_line))
        x_data1_red.append(np.array(x_data1_red_line))

  


    x_data1_right_blue = []
    x_data1_left_blue = []
    y_data1_blue = []
    




    for line_ in x_data1_blue:
        x_data1_right_blue.append(line_ + 0.06)
        x_data1_left_blue.append(line_ - 0.06)
        y_data1_blue.append(np.zeros(line_.shape[0]))



    x_data1_right_red = []
    x_data1_left_red = []
    y_data1_red = []

    for line_ in x_data1_red:
        x_data1_right_red.append(line_ + 0.06)
        x_data1_left_red.append(line_ - 0.06)
        y_data1_red.append(np.zeros(line_.shape[0]))






    y1_blue = np.zeros(len(x1_blue))
    y1_red = np.zeros(len(x1_red))



    scat1 = plt.scatter(np.array(x1_blue), y1_blue, s=80, c='blue')
    scat1_right = plt.scatter(np.array(x1_blue)+0.06, y1_blue, s=30, c='blue',marker="<")
    scat1_left = plt.scatter(np.array(x1_blue)-0.06, y1_blue, s=30, c='blue',marker=">")
    
    scat2 = plt.scatter(np.array(x1_red), y1_red, s=80, c='red')
    scat2_right = plt.scatter(np.array(x1_red)+0.06, y1_red, s=30, c='red',marker="<")
    scat2_left = plt.scatter(np.array(x1_red)-0.06, y1_red, s=30, c='red',marker=">")  


 
    for i in range(numframes):
        ani = animation.FuncAnimation(fig, update_plot_task2 ,frames = numframes ,fargs=(x_data1_blue,x_data1_right_blue,x_data1_left_blue,y_data1_blue, x_data1_red, y_data1_red,x_data1_right_red,x_data1_left_red,scat1,scat1_right,scat1_left, scat2,scat2_right,scat2_left))

    ani.save('plots/animation.gif', writer='imagemagick', fps=10)
    plt.show()



def plot_simulationL(data1,data2, L,name):


    numframes = len(data1)-1
    numpoints = len(data1[0])
    x1 = data1[0,:,0]
    y1 = data1[0,:,1]

    x2 = data2[0]

    x_data1 = data1[1:,:,0] 
    y_data1 = data1[1:,:,1]
    x_data1_right = x_data1 + 0.06
    x_data1_left = x_data1 - 0.06

    x_data2 = data2[1:] 

    fig = plt.figure()
    

    l, = plt.plot([], [], 'r-')
    plt.xlim(-0.5, L+0.5)
    plt.ylim(-1, 1)
    plt.xlabel('x')
    plt.title(name)


    line = np.linspace(0, L)
    wall = np.linspace(-3, 3)
    plt.plot(line, 0*line);
    plt.plot(0*wall,wall, color='black');
    plt.plot(L+0*wall, wall, color='black');



    scat1 = plt.scatter(x1, y1, s=80, c='blue')
    scat1_right = plt.scatter(x1+0.06, y1, s=30, c='blue',marker="<")
    scat1_left = plt.scatter(x1-0.06, y1, s=30, c='blue',marker=">")
        

    scat2 = plt.scatter(x2, y2, s=80, c='red')
    scat2_right = plt.scatter(x2+0.06, y1, s=30, c='red',marker="<")
    scat2_left = plt.scatter(x2-0.06, y1, s=30, c='red', marker=">")
 
    for i in range(numframes):
        ani = animation.FuncAnimation(fig, update_plot ,frames = numframes ,fargs=(x_data1,y_data1,x_data1_right,y_data1,x_data1_left,y_data1, x_data2,y_data2,x_data2_right,y_data2,x_data2_left,y_data2 ,scat1,scat1_right,scat1_left, scat2,scat2_right,scat2_left))

    ani.save('plots/animation.gif', writer='imagemagick', fps=10)
    plt.show()


def plot_simulationN(data_all, L, name):

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
    plt.xlim(-0.5, L+0.5)
    plt.ylim(-1, 1)
    plt.xlabel('x')
    plt.title(name)

    line = np.linspace(0, L)
    wall = np.linspace(-3, 3)
    plt.plot(line, 0*line);
    plt.plot(0*wall,wall);
    plt.plot(L+0*wall, wall);

    scats=[]
    colors = ['blue','red', 'brown', 'green']

    for i in range(len(data_all)):
        scats.append(plt.scatter(Xs[i], Ys[i], s=100, c=colors[i]))
  
    for i in range(numframes):
        ani = animation.FuncAnimation(fig, update_plotN ,frames = numframes ,fargs=(Xs_data,Ys_data,scats))

    ani.save('plots/animation.gif', writer='imagemagick', fps=10)
    plt.show()


def timeGraphOld(data1,data2, L,name, com = None):

    wall1=np.zeros((data1.shape[0],1))
    wall2=np.zeros((data1.shape[0],1))+L

    x_data1 = data1[0:,:,0] 
    y_data1 = data1[0:,:,1]

    x_data2 = data2[0:,:,0] 
    y_data2 = data2[0:,:,1]


    fig = plt.figure()

    base = pyplot.gca().transData
    rot = transforms.Affine2D().rotate_deg(90)

    plt.plot(x_data1,color='blue',label = 'optimal',transform= rot + base)
    plt.plot(x_data2,color='red',label = 'learned',transform= rot + base)
    plt.plot(np.concatenate((wall1, wall2), axis=1), color='black',transform = rot+base)

    custom_lines = [Line2D([0], [0], color='blue', lw=4),Line2D([0], [0], color='red', lw=4)]
    plt.legend(custom_lines, ['Optimal', 'Learned'],loc=4)

    plt.xlabel('x position')
    plt.ylabel('timesteps')
    plt.title(name)


    plt.show()  

def timeGraph(data1,data2, L,name, labels):

    x_data1 = data1[0:,:,0] 
    y_data1 = data1[0:,:,1]

    x_data2 = data2[0:,:,0]
    y_data2 = data2[0:,:,1]

    wall1=np.concatenate((np.arange(len(x_data1[:,0])).reshape(-1,1),np.zeros((data1.shape[0],1))), axis=1)
    wall2=np.concatenate((np.arange(len(x_data1[:,0])).reshape(-1,1),np.zeros((data1.shape[0],1))+L), axis=1)

    fig = plt.figure()

    plt.plot(wall1[:,1],wall1[:,0], color='black')
    plt.plot(wall2[:,1],wall2[:,0], color='black')

    for i in range(x_data2.shape[1]):
        line = np.concatenate((np.arange(len(x_data1[:,0])).reshape(-1,1),x_data1[:,i].reshape(-1,1)), axis=1)
        line2 = np.concatenate((np.arange(len(x_data2[:,0])).reshape(-1,1),x_data2[:,i].reshape(-1,1)), axis=1)
        plt.plot(line[:,1],line[:,0], color='blue')
      #  plt.plot(line2[:,1],line[:,0], color='red')
 
    #custom_lines = [Line2D([0], [0], color='blue', lw=4),Line2D([0], [0], color='red', lw=4)]
    #plt.legend(custom_lines, labels,loc=4)

    custom_lines = [Line2D([0], [0], color='blue', lw=4)]
    plt.legend(custom_lines, labels[0],loc=4)


    plt.xlabel('x position')
    plt.ylabel('timesteps')
    plt.title(name)

    plt.show()  




def timeGraphL2(data1,data2, L, name):

    x_data1 = data1[0:,:,0] 
    y_data1 = data1[0:,:,1]

    colors = np.zeros((data2.shape[0],data2.shape[1]))

    for i in range(colors.shape[0]):
        for j in range(colors.shape[1]):
            if data2[i,j]<0.5:
                colors[i,j] = 0.25
            else:
                colors[i,j]= -0.75

    colors = np.flip(colors, axis=0)



    #yticks =np.arange(colors.shape[0])
    yticks =np.arange(1,colors.shape[0]+1,colors.shape[0]//5)
    xticks =np.arange(colors.shape[1]) 


    fig, ax = plt.subplots()
    im = ax.imshow(colors, cmap="RdYlBu")

    # We want to show all ticks...
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    # ... and label them with the respective list entries
    ax.set_xticklabels(xticks+1)
    ax.set_yticklabels(np.flip(yticks))

 

    # Loop over data dimensions and create text annotations.
    #for i in range(len(timesteps)):
        #for j in range(len(agents)):
            #text = ax.text(j, i, colors[i, j],
            #           ha="center", va="center", color="w")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(colors.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(colors.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)


    ax.set_title("Output")
    fig.tight_layout()
    plt.ylabel("timesteps")
    plt.xlabel("Agent")
    plt.show()




def timeGraphL(data1,data2, L,name):

    x_data1 = data1[0:,:,0] 
    y_data1 = data1[0:,:,1]

    colors = np.zeros((data2.shape[0],data2.shape[1]))

    for i in range(colors.shape[0]):
        for j in range(colors.shape[1]):
            if data2[i,j]<0.5:
                colors[i,j] = -0.75
            else:
                colors[i,j]= 0.25

    wall1=np.concatenate((np.arange(len(x_data1[:,0])).reshape(-1,1),np.zeros((data1.shape[0],1))), axis=1)
    wall2=np.concatenate((np.arange(len(x_data1[:,0])).reshape(-1,1),np.zeros((data1.shape[0],1))+L), axis=1)

    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)

    plt.plot(wall1[:,1],wall1[:,0], color='black')
    plt.plot(wall2[:,1],wall2[:,0], color='black')

    norm = plt.Normalize(-L, L)

    for i in range(x_data1.shape[1]):
        points = np.array([x_data1[:,i], np.arange(len(x_data1[:,i]))]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        dydx = colors[:,i]



        lc = LineCollection(segments, cmap='inferno', norm=norm)
    
        # Set the values used for colormapping
        lc.set_array(dydx)
        lc.set_linewidth(2)
        line = axs.add_collection(lc)


   # fig.colorbar(line, ax=axs)

    axs.set_ylim(np.arange(len(x_data1[:,0])).min(), np.arange(len(x_data1[:,0])).max())
    axs.set_xlim(-0.2, L+0.2)
 
    plt.xlabel('x position')
    plt.ylabel('timesteps')
    plt.title(name)


    plt.show()  


def ComGraph(data1,data2, L,name, com):  #ComGraph
    x_data1 = data1[0:,:,0] 
    y_data1 = data1[0:,:,1]

    x_data2 = data2[0:,:,0]
    y_data2 = data2[0:,:,1]

    wall1=np.concatenate((np.arange(len(x_data1[:,0])).reshape(-1,1),np.zeros((data1.shape[0],1))), axis=1)
    wall2=np.concatenate((np.arange(len(x_data1[:,0])).reshape(-1,1),np.zeros((data1.shape[0],1))+L), axis=1)

    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)

    plt.plot(wall1[:,1],wall1[:,0], color='black')
    plt.plot(wall2[:,1],wall2[:,0], color='black')


    # Optimal lines
    #opt_lines=[]
    for i in range(x_data2.shape[1]):
       line = np.concatenate((np.arange(len(x_data1[:,0])).reshape(-1,1),x_data1[:,i].reshape(-1,1)), axis=1)
    #   plt.plot(line[:,1],line[:,0], color='blue')
    


    # Create a continuous norm to map from data points to colors
#    norm = plt.Normalize(-L, L)
    # solo per colori
    norm = plt.Normalize(0,1)

 

    #lines=[]

    for i in range(x_data2.shape[1]):
        points = np.array([x_data2[:,i], np.arange(len(x_data2[:,i]))]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        dydx = com[:,i]

        lc = LineCollection(segments, cmap='viridis', norm=norm)
    #    lc = LineCollection(segments, cmap='inferno', norm=norm)
    
        # Set the values used for colormapping
        lc.set_array(dydx)
        lc.set_linewidth(2)
        line = axs.add_collection(lc)
    #    lines.append(line)

    fig.colorbar(line, ax=axs)

    axs.set_ylim(np.arange(len(x_data2[:,0])).min(), np.arange(len(x_data2[:,0])).max())
    axs.set_xlim(-0.2, L+0.2)
 
    plt.xlabel('x position')
    plt.ylabel('timesteps')
    plt.xlabel("Agent")
    plt.title(name)


    plt.show()

def ComGraphL(data1,data2, L,name, com):  #ComGraphL

    x_data1 = data1[0:,:,0] 
    y_data1 = data1[0:,:,1]

    x_data2 = data2[0:,:,0]
    y_data2 = data2[0:,:,1]

    com= np.flip(com,0)


 
    #yticks =np.arange(colors.shape[0])
    yticks =np.arange(0,com.shape[0],com.shape[0]//5)
    xticks =np.arange(com.shape[1]) 

    fig, ax = plt.subplots()
    im = ax.imshow(com)

    # We want to show all ticks...
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    # ... and label them with the respective list entries
    ax.set_xticklabels(xticks+1)
    ax.set_yticklabels(np.flip(yticks))


    cbar = ax.figure.colorbar(im, ax=ax)
  #  cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Loop over data dimensions and create text annotations.
    #for i in range(len(timesteps)):
        #for j in range(len(agents)):
            #text = ax.text(j, i, colors[i, j],
            #           ha="center", va="center", color="w")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(com.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(com.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)


    ax.set_title("Communication")
    fig.tight_layout()
    plt.ylabel("timesteps")
    plt.xlabel("Agent")
    plt.show()


def timeGraphN(data_all, L, name):

    wall1=np.zeros((data_all[0].shape[0],1))
    wall2=np.zeros((data_all[0].shape[0],1))+L


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
    plt.plot(np.concatenate((wall1, wall2), axis=1), color='black',transform = rot+base)


    
    custom_lines = [Line2D([0], [0], color='blue', lw=4),Line2D([0], [0], color='red', lw=4),Line2D([0], [0], color='brown', lw=4),Line2D([0], [0], color='green', lw=4)]
    plt.legend(custom_lines, ['Optimal','Centralized' ,'Distributed','Learned'],loc=4)

    plt.xlabel('x position')
    plt.ylabel('timesteps')
    plt.title(name)

    plt.show()  

# update per i plot da 4
def update_plotN(i, Xs_data, Ys_data, scats):
    for n in range(len(Xs_data)):
        data = np.array([Xs_data[n][i],Ys_data[n][i]]).T
        scats[n].set_offsets(data)

    return scats[0], scats[1], scats[2], scats[3]



def update_plot_oooold(i, x_data1,y_data1, x_data2,y_data2, scat1, scat2):
    data1 = np.array([x_data1[i],y_data1[i]]).T
    scat1.set_offsets(data1)
    data2 = np.array([x_data2[i],y_data2[i]]).T
    scat2.set_offsets(data2)
    
    return scat1, scat2

def update_plot(i, x_data1,y_data1, x_data1_r, y_data1_r,x_data1_l,y_data1_l,x_data2,y_data2, x_data2_r,y_data2_r,x_data2_l,y_data2_l,scat1,scat1_r,scat1_l, scat2,  scat2_r, scat2_l):
    data1 = np.array([x_data1[i],y_data1[i]]).T
    scat1.set_offsets(data1)
    data2 = np.array([x_data2[i],y_data2[i]]).T
    scat2.set_offsets(data2)

    data1_r = np.array([x_data1_r[i],y_data1_r[i]]).T
    scat1_r.set_offsets(data1_r)
    data2_r = np.array([x_data2_r[i],y_data2_r[i]]).T
    scat2_r.set_offsets(data2_r)

    data1_l = np.array([x_data1_l[i],y_data1_l[i]]).T
    scat1_l.set_offsets(data1_l)
    data2_l = np.array([x_data2_l[i],y_data2_l[i]]).T
    scat2_l.set_offsets(data2_l)


    
    return scat1, scat2, scat1_r, scat2_r, scat1_l, scat2_l


def update_plot_task2(i, x_data1_blue,x_data1_right_blue,x_data1_left_blue,y_data1_blue, x_data1_red, y_data1_red,x_data1_right_red,x_data1_left_red,scat1,scat1_right,scat1_left, scat2,scat2_right,scat2_left):
    data1 = np.array([x_data1_blue[i],y_data1_blue[i]]).T
    scat1.set_offsets(data1)
    data2 = np.array([x_data1_red[i],y_data1_red[i]]).T
    scat2.set_offsets(data2)

    data1_r = np.array([x_data1_right_blue[i],y_data1_blue[i]]).T
    scat1_right.set_offsets(data1_r)
    data2_r = np.array([x_data1_right_red[i],y_data1_red[i]]).T
    scat2_right.set_offsets(data2_r)

    data1_l = np.array([x_data1_left_blue[i],y_data1_blue[i]]).T
    scat1_left.set_offsets(data1_l)
    data2_l = np.array([x_data1_left_red[i],y_data1_red[i]]).T
    scat2_left.set_offsets(data2_l)


    
    return scat1, scat2, scat1_right, scat2_right, scat1_left, scat2_left
        

def error_plot(errors, names):

#    rounder = lambda x: float("{0:.2f}".format(x))
#    vfunc = np.vectorize(rounder)
#    errors=vfunc(errors)

    colors = ['blue','yellow','green','red','purple']


    timesteps = np.linspace(0, len(errors[0]), num=len((errors[0])-1))

    plt.grid(True, which='both')

    # Linear X axis, Logarithmic Y axis
    for i in range(len(errors)):
        plt.plot(timesteps, errors[i][:,0] , label=names[i])
    plt.legend(loc=2)

    for i in range(len(errors)):
        plt.fill_between(x =timesteps, y1=errors[i][:,1],y2=errors[i][:,2], facecolor=colors[i],color=colors[i], alpha=0.2)

    plt.title('Error for every net')
    plt.xlabel('Timestep')
    plt.ylabel('Error')

    plt.show()


        
