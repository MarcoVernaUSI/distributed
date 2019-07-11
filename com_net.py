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

        print('Building the Graph')
        ##### First Hidden layer

        self.W_h = tf.Variable(tf.truncated_normal([input_size+com_size,64], stddev=0.1), name = 'W_h') 
        self.b_h = tf.Variable(tf.truncated_normal([64], stddev=0.1), name = 'b_h')
        
        self.W_h2 = tf.Variable(tf.truncated_normal([input_size+com_size,64], stddev=0.1), name = 'W_h2') 
        self.b_h2 = tf.Variable(tf.truncated_normal([64], stddev=0.1), name = 'b_h2')
        
        ##### Second Hidden layer

        self.W_hh = tf.Variable(tf.truncated_normal([64,32], stddev=0.1), name = 'W_hh') 
        self.b_hh = tf.Variable(tf.truncated_normal([32], stddev=0.1), name = 'b_hh')
        
        self.W_hh2 = tf.Variable(tf.truncated_normal([64,32], stddev=0.1), name = 'W_hh2') 
        self.b_hh2 = tf.Variable(tf.truncated_normal([32], stddev=0.1), name = 'b_hh2')
        
        ##### Output layer

        self.W = tf.Variable(tf.truncated_normal([32, output_size], stddev=0.1), name = 'W')
        self.b = tf.Variable(tf.truncated_normal([output_size], stddev=0.1), name = 'b')

        self.W_c = tf.Variable(tf.truncated_normal([32, output_size], stddev=0.1), name = 'W_c') 
        self.b_c = tf.Variable(tf.truncated_normal([output_size], stddev=0.1), name = 'b_c')

        ##### Data placeholder

        self.X = tf.placeholder(tf.float32, [self.batch_size * self.sequence_length, N,input_size]) 
        self.Y = tf.placeholder(tf.float32, [self.batch_size * self.sequence_length, N]) ###
        self.Com = tf.placeholder(tf.float32, [N,com_size])
        self.inputs_series = tf.unstack(self.X, axis=0)

        ######################### 
        # Forward pass

        self.current_com = self.Com
        self.outputs_series = []

        for idx, current_input in enumerate(self.inputs_series):

            input_and_com_concatenated = tf.concat([current_input, self.current_com],1)  # Increasing number of columns

            h1, h2 = self.forward(input_and_com_concatenated)

            self.output_control = tf.matmul(h1, self.W) + self.b 
            self.outputs_series.append(self.output_control)

            if ((idx+1)%self.sequence_length) == 0:
                next_com = tf.zeros([N,com_size], tf.float32)
            else:
                output_com = tf.matmul(h2, self.W_c) + self.b_c  # Broadcasted addition
                next_com = self.transmit(output_com)
            self.current_com = next_com

        output_s = tf.stack(self.outputs_series)
        output_s = tf.reshape(output_s, [output_s.shape[0],-1])

        self.losses = tf.reduce_mean(tf.square(self.Y - output_s)) 
        self.total_loss = tf.reduce_mean(self.losses)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)

        ################################
        # Prediction

        self.new_sample = tf.placeholder(tf.float32, [N,input_size])
        self.Com_sample = tf.placeholder(tf.float32, [N,com_size])
        self.pred_input = tf.concat([self.new_sample, self.Com_sample],1, name='p_input')  # Increasing number of columns
        
        h1p, h2p = self.forward(self.pred_input)

        output_com_p = tf.matmul(h2p, self.W_c) + self.b_c  # Broadcasted addition
        self.output_control_p = tf.matmul(h1p, self.W) + self.b 

        self.next_com_p = self.transmit(output_com_p)

        print('Building ended')
        ################################

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

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

    def prepare_data(self, net_inputs, net_outputs):
        t_data= np.array(net_inputs)
        targets= np.array(net_outputs)
        t_data=t_data[:,:,:,3:5]

        t_data = t_data.reshape(-1, t_data.shape[2], t_data.shape[3])
        targets = targets.reshape(-1, targets.shape[2])
        self.len_dataset = t_data.shape[0]
        return t_data, targets

    def train(self, data, targets, learning_rate = 0.002, buffer_epoch = 25):
        ## Da implementare early stopping

        self.learning_rate = learning_rate
        best_model = 10

        val_data = data[round(data.shape[0]*0.8):]
        t_data = data[:round(data.shape[0]*0.8)]
        val_targets = targets[round(data.shape[0]*0.8):]
        t_targets = targets[:round(data.shape[0]*0.8)]
        len_dataset = t_data.shape[0]
        len_validation = val_data.shape[0]

        init = tf.initialize_all_variables()
        self.sess.run(init)
        
        loss_list=[]
        for epoch_idx in range(self.epochs):
            print('epoch',epoch_idx)
            _current_com = np.zeros((self.N,self.com_size))
            num_batches = len_dataset//self.sequence_length//self.batch_size

            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size * self.sequence_length
                end_idx = start_idx + self.batch_size * self.sequence_length

                batchX = t_data[start_idx:end_idx,:,:]
                batchY = t_targets[start_idx:end_idx,:]

                ###############################
                _total_loss, _train_step, _current_com, _outputs_series = self.sess.run(
                    [self.total_loss, self.train_step, self.current_com, self.outputs_series],
                    feed_dict={
                        self.X:batchX,
                        self.Y:batchY,
                        self.Com: _current_com
                    })

                loss_list.append(_total_loss)

            print("training Loss", np.mean(np.array(_total_loss)))


            #### Validation 
            _current_com = np.zeros((self.N,self.com_size))
            num_batches = len_validation//self.sequence_length//self.batch_size

            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size * self.sequence_length
                end_idx = start_idx + self.batch_size * self.sequence_length

                batchX = val_data[start_idx:end_idx,:,:]
                batchY = val_targets[start_idx:end_idx,:]

            _total_loss_val = self.sess.run(
                    [self.total_loss],
                    feed_dict={
                        self.X:batchX,
                        self.Y:batchY,
                        self.Com: _current_com
                    })
            
            print("Validation Loss", np.mean(np.array(_total_loss_val)))

            # Saving Model
            if np.mean(np.array(_total_loss_val)) < best_model and epoch_idx >= buffer_epoch:
                print('Saving model')
                save_path = self.saver.save(self.sess, "model/model.ckpt")
                best_model = _total_loss_val
                print("Model saved in path: %s" % save_path)

        return _total_loss, _total_loss_val

    def predict(self, data, comm):

        data=data.astype(np.float32)
        control, transmission = self.sess.run([self.output_control_p, self.next_com_p], feed_dict={self.new_sample:data,self.Com_sample:comm})
        return control, transmission

    def forward(self, input_and_com_concatenated):
        h1 = tf.nn.tanh(tf.matmul(input_and_com_concatenated, self.W_h)+self.b_h)
        h2= tf.nn.tanh(tf.matmul(input_and_com_concatenated, self.W_h2)+self.b_h2)

        hh1= tf.nn.tanh(tf.matmul(h1, self.W_hh)+self.b_hh)
        hh2=  tf.nn.tanh(tf.matmul(h2, self.W_hh2)+self.b_hh2)

    #   hhh1= tf.nn.tanh(tf.matmul(hh1, self.W_hhh)+self.b_hhh)
    #   hhh2=  tf.nn.tanh(tf.matmul(hh2, self.W_hhh2)+self.b_hhh2)   

        return hh1, hh2     

##########################################################################################