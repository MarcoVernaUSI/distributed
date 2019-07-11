####################
# Neural net, simple layer net
####################

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np





class sNet:
    def __init__ (self, N, input_size, com_size, output_size ,sequence_length = 100):

        self.N = N
        self.input_size = input_size
        self.com_size = com_size
        self.output_size = output_size
        self.sequence_length = sequence_length

        ##### Data placeholder

        self.current_input = tf.placeholder(tf.float32, [self.N,input_size])
        self.current_com = tf.placeholder(tf.float32, [N,com_size])
        self.idx = tf.placeholder(tf.int32)


        ##### First Hidden layer

        self.W_h = tf.placeholder(tf.float32, [input_size+com_size,32]) 
        self.b_h = tf.placeholder(tf.float32, [32])

        self.W_h2 = tf.placeholder(tf.float32, [input_size+com_size,32]) 
        self.b_h2 = tf.placeholder(tf.float32, [32])
        
        ##### Second Hidden layer

        self.W_hh = tf.placeholder(tf.float32, [64,32]) 
        self.b_hh = tf.placeholder(tf.float32, [32])
        
        self.W_hh2 = tf.placeholder(tf.float32, [64,32]) 
        self.b_hh2 = tf.placeholder(tf.float32, [32])
        
        ##### Output layer

        self.W = tf.placeholder(tf.float32, [32, output_size])
        self.b = tf.placeholder(tf.float32, [output_size])

        self.W_c = tf.placeholder(tf.float32, [32, output_size]) 
        self.b_c = tf.placeholder(tf.float32, [output_size])

        ######################### 
        # Forward pass
 
        self.input_and_com_concatenated = tf.concat([self.current_input, self.current_com],1)  # Increasing number of columns

        h1, h2 = self.forward(self.input_and_com_concatenated)

        self.output_control = tf.matmul(h1, self.W) + self.b 
      
        if ((self.idx+1)%self.sequence_length) == 0:
            self.next_com = tf.zeros([N,com_size], tf.float32)
        else:
            self.output_com = tf.matmul(h2, self.W_c) + self.b_c  # Broadcasted addition
            self.next_com = self.transmit(self.output_com)
        
        ################################
        # Prediction

        self.new_sample = tf.placeholder(tf.float32, [N,input_size])
        self.Com_sample = tf.placeholder(tf.float32, [N,com_size])

        self.input_and_com_concatenated_p = tf.concat([self.new_sample, self.Com_sample],1, name='p_input')  # Increasing number of columns
        
        h1p, h2p = self.forward(self.input_and_com_concatenated)
        #h = tf.nn.tanh(tf.matmul(input_and_com_concatenated, self.W_h)+self.b_h)

        self.output_com_p = tf.matmul(h2, self.W_c) + self.b_c  # Broadcasted addition
        self.output_control_p = tf.matmul(h1, self.W) + self.b    #questo poi porovo a spostarlo fuori 

        self.next_com_p = self.transmit(self.output_com)

        ################################
        self.sess = None

    def transmit(self, comunication):
        l = int(comunication.shape[0])-1
        new_com = np.zeros((int(comunication.shape[0]),int(comunication.shape[1])*2), dtype=np.float32)
        first_column = [tf.zeros([1], tf.float32),0,0,0,0]   # poi va cambiato (non dipende dal numero di agenti)
        second_column = [0,0,0,0,tf.zeros([1], tf.float32)]

        for i in range(l):
            first_column[i+1]=comunication[i]
            second_column[i]=comunication[i+1]

        fc = tf.stack(first_column)
        sc = tf.stack(second_column)

        new_com = tf.stack([fc,sc],axis=1)
        new_com = tf.reshape(new_com, [new_com.shape[0],new_com.shape[1]])

        return new_com

    def run_net(self, idx, current_input, current_com, W_h,b_h,W_h2,b_h2,W_hh,b_hh,W_hh2,b_hh2,W,b,W_c,b_c):
                
        _output_control, _next_com = self.sess.run(
            [self.output_control, self.next_com],
            feed_dict={
                self.idx:idx,
                self.current_input: current_input,
                self.current_com: current_com,
                self.W_h:W_h, 
                self.b_h:b_h,
                self.W_h2:W_h2,
                self.b_h2:b_h2,
                self.W_hh:W_hh, 
                self.b_hh:b_hh,
                self.W_hh2:W_hh2, 
                self.b_hh2:b_hh2,
                self.W:W,
                self.b:b,
                self.W_c:W_c, 
                self.b_c:b_c,
            })

        return _outputs_control, _next_com

    def predict(self, data, comm, W_h,b_h,W_h2,b_h2,W_hh,b_hh,W_hh2,b_hh2,W,b,W_c,b_c):

        control, transmission = self.sess.run([self.output_control, self.next_com], 
            feed_dict={self.new_sample:data,self.Com_sample:comm,self.W_h:W_h, 
                self.b_h:b_h,
                self.W_h2:W_h2,
                self.b_h2:b_h2,
                self.W_hh:W_hh, 
                self.b_hh:b_hh,
                self.W_hh2:W_hh2, 
                self.b_hh2:b_hh2,
                self.W:W,
                self.b:b,
                self.W_c:W_c, 
                self.b_c:b_c,})

        return control, transmission

    def forward(self, input_and_com_concatenated):
        h1 = tf.nn.tanh(tf.matmul(input_and_com_concatenated, self.W_h)+self.b_h)
        h2= tf.nn.tanh(tf.matmul(input_and_com_concatenated, self.W_h2)+self.b_h2)

    #   hh1= tf.nn.tanh(tf.matmul(h1, self.W_hh)+self.b_hh)
    #   hh2=  tf.nn.tanh(tf.matmul(h2, self.W_hh2)+self.b_hh2) 

        return h1, h2     

##########################################################################################