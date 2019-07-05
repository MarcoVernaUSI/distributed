####################
# Neural net, simple layer net
####################

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class CNet:
    def __init__(self, input_size, epochs=100, batch_size=100, truncated_backprop_length=10):
        self.input_size = input_size
        self.output_size = 1
        self.epochs = epochs
        self.communcation_size = 1
        self.truncated_backprop_length = truncated_backprop_length         ##??????
        self.batch_size = batch_size
        self.len_dataset=0
        self.learning_rate=0.03

        self.batchX_placeholder = tf.placeholder(tf.float32, [self.batch_size, self.truncated_backprop_length, self.input_size])
        self.batchY_placeholder = tf.placeholder(tf.float32, [self.batch_size, self.truncated_backprop_length])

        self.comm = tf.placeholder(tf.float32, [self.batch_size, self.communcation_size, self.input_size])

        self.new_sample = tf.placeholder(tf.float32,[None,self.input_size+(self.input_size*self.communcation_size)])


        # pesi per il nuovo stato
        self.W = tf.Variable(np.random.rand(self.input_size+(self.input_size*self.communcation_size), (self.input_size*self.communcation_size)), dtype=tf.float32)  # inizializzare con normale, flatto tutto.
        self.b = tf.Variable(np.zeros((self.batch_size,(self.input_size*self.communcation_size))), dtype=tf.float32)             ## inizializzare con normale



        self.W2 = tf.Variable(np.random.rand(self.input_size+(self.input_size*self.communcation_size), self.output_size),dtype=tf.float32) ## inizializzare con normale
        self.b2 = tf.Variable(np.zeros((self.batch_size, self.output_size)), dtype=tf.float32)   ## inizializzare con normale

        # unpack columns
        self.inputs_series = tf.unstack(self.batchX_placeholder, axis=1)
        self.labels_series = tf.unstack(self.batchY_placeholder, axis=1)

        self.sess = None

        # Forward
        self.current_state = self.comm
        states_series = []
        # qua prendo il numero di input pari al numero di agenti, lo faccio N volte
        for current_input in self.inputs_series:

            current_input = tf.reshape(current_input, [self.batch_size, self.input_size])
            self.current_state = tf.reshape(self.current_state, [self.batch_size, -1])

            # Questo deve prendere le comunicazioni delle altre reti
            input_and_state_concatenated = tf.concat([current_input, self.current_state],1)  # Increasing number of columns

            next_state = tf.tanh(tf.matmul(input_and_state_concatenated, self.W) + self.b)  # Broadcasted addition
            next_state = tf.reshape(next_state, [self.batch_size, self.communcation_size,self.input_size])

            states_series.append(input_and_state_concatenated)
            self.current_state = next_state


        self.nets = [tf.matmul(state, self.W2) + self.b2 for state in states_series]
        self.prediction = tf.matmul(self.new_sample, self.W2) + self.b2

        self.losses = [tf.reduce_mean(tf.square(tf.reshape(expected,[expected.shape[0],-1]) - net)) for expected, net in zip(self.labels_series, self.nets)]


        self.total_loss = tf.reduce_mean(self.losses)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)



    def prepare_data(self, net_inputs, net_outputs):    # Per ora prendo i dati da un singolo agente
        t_data= np.array(net_inputs)
        targets= np.array(net_outputs)
        t_data=t_data[:,:,0,3:5] # Per ora prendo i dati da un singolo agente

      #  t_data = t_data.reshape(t_data.shape[0]*t_data.shape[1], -1)
      #  targets = targets.reshape(targets.shape[0]*targets.shape[1], -1)
      #  targets= targets[:,0] # dati da un singolo agente
      #  self.len_dataset = len(t_data)

        t_data = t_data.reshape(self.batch_size, -1, t_data.shape[2])
        targets = targets.reshape(self.batch_size, -1, targets.shape[2])
        targets= targets[:,:,0] # dati da un singolo agente
        self.len_dataset = t_data.shape[0]*t_data.shape[1]
      

        return t_data, targets

    def train(self, data, targets, learning_rate=0.03):

        self.learning_rate = learning_rate
        self.sess = tf.Session()
        #with tf.Session() as self.sess:
        self.sess.run(tf.initialize_all_variables())
        loss_list = []

        for epoch_idx in range(self.epochs):
            _current_state = np.zeros((self.batch_size, self.communcation_size, self.input_size))

            #print("New data, epoch", epoch_idx)

            num_batches = self.len_dataset//self.batch_size//self.truncated_backprop_length

            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.truncated_backprop_length
                end_idx = start_idx + self.truncated_backprop_length

                batchX = data[:,start_idx:end_idx,:]
                batchY = targets[:,start_idx:end_idx]

                _total_loss, _train_step, _current_state, _nets = self.sess.run(
                    [self.total_loss, self.train_step, self.current_state, self.nets],
                    feed_dict={
                        self.batchX_placeholder:batchX,
                        self.batchY_placeholder:batchY,
                        self.comm:_current_state
                    })

                loss_list.append(_total_loss)

                if batch_idx%10 == 0:
                    print("Step",batch_idx, "Loss", _total_loss)
        return _nets

    def predict(self, data):
        self.truncated_backprop_length = 1
        self.batch_size = 1

        data = data[0]

        data = np.pad(data, (0,2), 'constant', constant_values = (0,0))
        data = data.reshape(-1, data.shape[0])
        actual = self.sess.run(self.prediction, feed_dict={self.new_sample:data})
        return actual

    
