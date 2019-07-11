####################
# Neural net, simple layer net
####################

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np





class ComNet:
    def __init__ (self, N, input_size, com_size, output_size, epochs = 100, batch_size = 10, sequence_length = 100, n_simulation = 120):

        self.epochs = epochs
        self.len_dataset = 0
        self.N = N
        self.input_size = input_size
        self.com_size = com_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.learning_rate=0.03
        self.sequence_length = sequence_length

        self.W_h = tf.Variable(tf.truncated_normal([input_size+com_size,64], stddev=0.1), name = 'W_h') 
        self.b_h = tf.Variable(tf.truncated_normal([64], stddev=0.1), name = 'b_h')
        
    #    self.W_h2 = tf.Variable(tf.truncated_normal([input_size+com_size,32], stddev=0.1), name = 'W_h2') 
    #    self.b_h2 = tf.Variable(tf.truncated_normal([32], stddev=0.1), name = 'b_h2')
        

        

        self.W_hh = tf.Variable(tf.truncated_normal([64,32], stddev=0.1), name = 'W_hh') 
        self.b_hh = tf.Variable(tf.truncated_normal([32], stddev=0.1), name = 'b_hh')
        
    #    self.W_hh2 = tf.Variable(tf.truncated_normal([64,32], stddev=0.1), name = 'W_hh2') 
    #    self.b_hh2 = tf.Variable(tf.truncated_normal([32], stddev=0.1), name = 'b_hh2')
        

     #   self.W_hhh = tf.Variable(tf.truncated_normal([16,32], stddev=0.1), name = 'W_hhh') 
     #   self.b_hhh = tf.Variable(tf.truncated_normal([32], stddev=0.1), name = 'b_hhh')
        
     #   self.W_hhh2 = tf.Variable(tf.truncated_normal([16,32], stddev=0.1), name = 'W_hhh2') 
     #   self.b_hhh2 = tf.Variable(tf.truncated_normal([32], stddev=0.1), name = 'b_hhh2')
        


        self.W = tf.Variable(tf.truncated_normal([32, output_size], stddev=0.1), name = 'W')
        self.b = tf.Variable(tf.truncated_normal([output_size], stddev=0.1), name = 'b')

        self.W_c = tf.Variable(tf.truncated_normal([32, output_size], stddev=0.1), name = 'W_c') 
        self.b_c = tf.Variable(tf.truncated_normal([output_size], stddev=0.1), name = 'b_c')


        self.X = tf.placeholder(tf.float32, [self.batch_size * self.sequence_length, N,input_size]) 
        self.Y = tf.placeholder(tf.float32, [self.batch_size * self.sequence_length, N]) ###
        self.Com = tf.placeholder(tf.float32, [N,com_size])

        #self.batch_size = tf.shape(self.X)[0]
        self.new_sample = tf.placeholder(tf.float32, [N,input_size])
        self.Com_sample = tf.placeholder(tf.float32, [N,com_size])
        

        self.inputs_series = tf.unstack(self.X, axis=0)
        self.labels_series = tf.unstack(self.Y, axis=0)

        ######################### forward
        self.current_com = self.Com

        self.sess = tf.Session()

        self.outputs_series = []

        for idx, current_input in enumerate(self.inputs_series):
        
            input_and_com_concatenated = tf.concat([current_input, self.current_com],1)  # Increasing number of columns
        #    h1 = tf.nn.tanh(tf.matmul(input_and_com_concatenated, self.W_h)+self.b_h)
        #    h2= tf.nn.tanh(tf.matmul(input_and_com_concatenated, self.W_h2)+self.b_h2)
            h = self.forward(input_and_com_concatenated)
     # 

            output_com = tf.matmul(h, self.W_c) + self.b_c  # Broadcasted addition
            self.output_control = tf.matmul(h, self.W) + self.b 

            self.outputs_series.append(self.output_control)

            if ((idx+1)%self.sequence_length) == 0:
                next_com = tf.zeros([N,com_size], tf.float32)
            else:
                next_com = self.transmit(output_com)
            self.current_com = next_com

        self.losses = [tf.reduce_mean(tf.square(tf.reshape(expected,[expected.shape[0],-1]) - net)) for expected, net in zip(self.labels_series, self.outputs_series)]
        self.total_loss = tf.reduce_mean(self.losses)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)

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

    def prepare_data(self, net_inputs, net_outputs):    # Per ora prendo i dati da un singolo agente
        t_data= np.array(net_inputs)
        targets= np.array(net_outputs)
        t_data=t_data[:,:,:,3:5]

        t_data = t_data.reshape(-1, t_data.shape[2], t_data.shape[3])
        targets = targets.reshape(-1, targets.shape[2])
        self.len_dataset = t_data.shape[0]
        return t_data, targets

    def train(self, data, targets, learning_rate = 0.03):
        ## Da implementare early stopping

        self.learning_rate = learning_rate

        init = tf.initialize_all_variables()
        self.sess.run(init)
        
        loss_list=[]
        for epoch_idx in range(self.epochs):
            print('epoch',epoch_idx)
            _current_com = np.zeros((self.N,self.com_size))
            num_batches = self.len_dataset//self.sequence_length//self.batch_size

            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size * self.sequence_length
                end_idx = start_idx + self.batch_size * self.sequence_length

                batchX = data[start_idx:end_idx,:,:]
                batchY = targets[start_idx:end_idx,:]

                ###############################
                _total_loss, _train_step, _current_com, _outputs_series = self.sess.run(
                    [self.total_loss, self.train_step, self.current_com, self.outputs_series],
                    feed_dict={
                        self.X:batchX,
                        self.Y:batchY,
                        self.Com: _current_com
                    })

                loss_list.append(_total_loss)

                if batch_idx%100 == 0:
                    print("Loss", _total_loss)
        return _outputs_series

    def predict(self, data, comm):

        data=data.astype(np.float32)


        input_and_com_concatenated = tf.concat([self.new_sample, self.Com_sample],1, name='p_input')  # Increasing number of columns
        
        h = self.forward(input_and_com_concatenated)
        #h = tf.nn.tanh(tf.matmul(input_and_com_concatenated, self.W_h)+self.b_h)

        output_com = tf.matmul(h, self.W_c) + self.b_c  # Broadcasted addition
        output_control = tf.matmul(h, self.W) + self.b    #questo poi porovo a spostarlo fuori 

        next_com = self.transmit(output_com)

        control, transmission = self.sess.run([output_control, next_com], feed_dict={self.new_sample:data,self.Com_sample:comm})

        return control, transmission

        

    def forward(self, input_and_com_concatenated):
        h1 = tf.nn.tanh(tf.matmul(input_and_com_concatenated, self.W_h)+self.b_h)
     #   h2= tf.nn.tanh(tf.matmul(input_and_com_concatenated, self.W_h2)+self.b_h2)

        hh1= tf.nn.tanh(tf.matmul(h1, self.W_hh)+self.b_hh)
    #    hh2=  tf.nn.tanh(tf.matmul(h2, self.W_hh2)+self.b_hh2)

     #   hhh1= tf.nn.tanh(tf.matmul(hh1, self.W_hhh)+self.b_hhh)
     #   hhh2=  tf.nn.tanh(tf.matmul(hh2, self.W_hhh2)+self.b_hhh2)   

        return hh1     

#########################################################################################